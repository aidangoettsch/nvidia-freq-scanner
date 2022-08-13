use crate::gui::Message::{SelectGpu, UpdateLowerBound, UpdateUpperBound, UpdateInitialStep, StartTest, TestUpdate};
use iced::{executor, pick_list, time, Align, Application, Column, Command, Element, PickList, Row, Subscription, Text, Rectangle, Canvas, Container, Length, TextInput, text_input, Button, button};
use num::Num;
use nvapi_hi::{Gpu, KilohertzDelta, GpuInfo, GpuStatus, Kibibytes, ClockDomain};
use std::fmt::Display;
use crate::number_input::NumberInput;
use crate::number_input;
use num::traits::NumAssignOps;
use std::str::FromStr;
use iced::canvas::{Program, Cursor, Geometry, Cache};
use crate::plotters_iced::IcedBackend;
use plotters::prelude::{IntoFont, EmptyElement, Circle, PointSeries, RED, LineSeries, ChartBuilder, IntoDrawingArea, WHITE, GREEN, BLUE, YELLOW};
use plotters::prelude::Text as PlottersText;
use nvapi_hi::nvapi::sys::gpu::clock::PublicClockId;
use nvapi_hi::nvapi::sys::gpu::pstate::PstateId;
use crate::test::{Progress, TestResultPoint, Test};

#[derive(Debug, Default)]
struct GraphCtx {
    cache: Cache,
    results: Vec<TestResultPoint>,
    x_low_bound: KilohertzDelta,
    x_high_bound: KilohertzDelta,
}

#[derive(Debug)]
pub(crate) struct TestGui {
    gpu_list: Vec<DeviceChoice>,
    gpu_idx: usize,
    test_running: bool,
    dropdown_state: pick_list::State<DeviceChoice>,
    memory_size_state: OptionalCellState,
    bus_width_state: OptionalCellState,
    core_clock_state: OptionalCellState,
    memory_clock_state: OptionalCellState,
    memory_type_state: text_input::State,
    lower_bound_state: number_input::State,
    upper_bound_state: number_input::State,
    initial_step_state: number_input::State,
    test_button_state: button::State,
    bound_lower: KilohertzDelta,
    bound_upper: KilohertzDelta,
    initial_step: KilohertzDelta,
    graph_ctx: GraphCtx,
    gpu_info: Option<GpuInfo>,
    gpu_status: Option<GpuStatus>,
}

#[derive(Debug, Clone)]
pub(crate) enum Message {
    Tick(),
    SelectGpu(usize),
    UpdateLowerBound(i32),
    UpdateUpperBound(i32),
    UpdateInitialStep(i32),
    StartTest,
    TestUpdate(Progress),
}

#[derive(Debug, Eq, PartialEq, Clone)]
struct DeviceChoice {
    name: String,
    idx: usize,
    mem_clock_limit_lower: KilohertzDelta,
    mem_clock_limit_upper: KilohertzDelta,
}

impl Display for DeviceChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.name, self.idx)
    }
}

#[derive(Default, Debug)]
struct OptionalCellState {
    number_state: number_input::State,
    text_state: text_input::State,
}

impl TestGui {
    fn text_cell<'a> (
        state: &'a mut text_input::State,
        title: &'a str,
        value: String,
    ) -> Column<'a, <TestGui as Application>::Message> {
        Column::with_children(vec![
            Text::new(title).into(),
            TextInput::new(state, &value, &value, |_| Message::Tick())
                .padding(5)
                .width(Length::Units(127))
                .into(),
        ])
            .align_items(Align::Center)
            .spacing(2)
    }

    fn number_cell<'a, T> (
        state: &'a mut number_input::State,
        title: &'a str,
        value: T,
        unit_hint: &'a str,
    ) -> Column<'a, <TestGui as Application>::Message> where
        T: 'static + Num + NumAssignOps + PartialOrd + Display + FromStr + Copy,
    {
        Column::with_children(vec![
            Text::new(title).into(),
            NumberInput::new(state, value, value, unit_hint.to_string(), false, |_| Message::Tick()).into()
        ])
            .align_items(Align::Center)
            .spacing(2)
    }

    fn mutable_number_cell<'a, T, F> (
        state: &'a mut number_input::State,
        title: &'a str,
        value: T,
        range: (T, T),
        unit_hint: &'a str,
        on_changed: F
    ) -> Column<'a, <TestGui as Application>::Message> where
        T: 'static + Num + NumAssignOps + PartialOrd + Display + FromStr + Copy,
        F: 'static + Fn(T) -> Message + Copy,
    {
        Column::with_children(vec![
            Text::new(title).into(),
            NumberInput::new(state, value, range.1, unit_hint.to_string(), true, on_changed)
                .bounds(range)
                .into()
        ])
            .align_items(Align::Center)
            .spacing(2)
    }

    fn option_text_cell<'a> (
        state: &'a mut text_input::State,
        title: &'a str,
        value: Option<String>,
    ) -> Column<'a, <TestGui as Application>::Message> {
        match value {
            Some(v) => Self::text_cell(state, title, v),
            None => Self::text_cell(state, title, "".to_string())
        }
    }

    fn option_number_cell<'a, T> (
        state: &'a mut OptionalCellState,
        title: &'a str,
        value: Option<T>,
        unit_hint: &'a str,
    ) -> Column<'a, <TestGui as Application>::Message> where
        T: 'static + Num + NumAssignOps + PartialOrd + Display + FromStr + Copy,
    {
        match value {
            Some(v) => Self::number_cell(&mut state.number_state, title, v, unit_hint),
            None => Self::text_cell(&mut state.text_state, title, "".to_string())
        }
    }
}

impl Application for TestGui {
    type Executor = executor::Default;
    type Message = Message;
    type Flags = ();

    fn new(_flags: ()) -> (TestGui, Command<Self::Message>) {
        let gpu_list = Gpu::enumerate()
            .unwrap_or(vec![])
            .iter()
            .enumerate()
            .map(|(idx, gpu)| {
                let info = gpu
                    .info()
                    .expect(&format!("Could not get info for GPU {}", idx));

                let p0_delta_limits = info.pstate_limits[&PstateId::P0][&ClockDomain::Memory].frequency_delta.expect(&format!("Could not get frequency limits for GPU {}", idx));

                DeviceChoice {
                    idx,
                    name: format!("{} {}", info.vendor, info.name),
                    mem_clock_limit_lower: p0_delta_limits.min,
                    mem_clock_limit_upper: p0_delta_limits.max,
                }
            })
            .collect::<Vec<DeviceChoice>>();

        let bound_upper = gpu_list[0].mem_clock_limit_upper;

        (
            TestGui {
                gpu_list,
                gpu_idx: 0,
                test_running: false,
                dropdown_state: Default::default(),
                memory_size_state: Default::default(),
                bus_width_state: Default::default(),
                core_clock_state: Default::default(),
                memory_clock_state: Default::default(),
                memory_type_state: Default::default(),
                lower_bound_state: Default::default(),
                upper_bound_state: Default::default(),
                initial_step_state: Default::default(),
                test_button_state: Default::default(),
                bound_lower: KilohertzDelta(0),
                bound_upper,
                initial_step: KilohertzDelta(50000),
                graph_ctx: GraphCtx {
                    cache: Default::default(),
                    results: vec![],
                    x_low_bound: KilohertzDelta(0),
                    x_high_bound: bound_upper,
                },
                gpu_info: None,
                gpu_status: None
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("NVIDIA Clock Scanner")
    }

    fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
        match message {
            SelectGpu(idx) => {
                self.gpu_idx = idx;

                self.bound_lower = KilohertzDelta(0);
                self.bound_upper = self.gpu_list[self.gpu_idx].mem_clock_limit_upper
            }
            UpdateLowerBound(bound_lower) => {
                self.bound_lower = KilohertzDelta(bound_lower * 1000);
                self.graph_ctx.x_low_bound = self.bound_lower
            }
            UpdateUpperBound(bound_upper) => {
                self.bound_upper = KilohertzDelta(bound_upper * 1000);
                self.graph_ctx.x_high_bound = self.bound_upper
            }
            UpdateInitialStep(initial_step) => {
                self.initial_step = KilohertzDelta(initial_step * 1000)
            }
            StartTest => {
                self.graph_ctx.results = vec![];
                self.test_running = true
            }
            TestUpdate(progress) => {
                match progress {
                    Progress::Finished => {
                        self.test_running = false
                    }
                    Progress::Result(point) => {
                        self.graph_ctx.results.push(point);
                        self.graph_ctx.cache.clear()
                    }
                    Progress::Errored(err) => {
                        println!("{:?}", err);
                        self.test_running = false
                    }
                };
            }
            _ => {}
        };

        let gpus = Gpu::enumerate().unwrap_or_default();
        self.gpu_info = match gpus[self.gpu_idx as usize].info() {
            Ok(info) => Some(info),
            Err(_) => None
        };

        self.gpu_status = match gpus[self.gpu_idx as usize].status() {
            Ok(status) => Some(status),
            Err(_) => None
        };

        Command::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        Subscription::batch(vec![
            time::every(std::time::Duration::from_millis(500)).map(|_| Message::Tick()),
            if self.test_running {
                match Test::new(self.gpu_idx, self.bound_lower, self.bound_upper, self.initial_step) {
                    Ok(s) => iced::Subscription::from_recipe(s).map(Message::TestUpdate),
                    Err(_) => Subscription::none()
                }
            } else {
                Subscription::none()
            },
        ])
    }

    fn view(&mut self) -> Element<Self::Message> {
        let footer = Row::with_children(vec![PickList::new(
            &mut self.dropdown_state,
            &self.gpu_list,
            Some(self.gpu_list[0].clone()),
            |device| SelectGpu(device.idx),
        )
        .into()])
        .align_items(Align::Center);

        let graph = Canvas::new(&self.graph_ctx)
            .width(Length::Units(1000))
            .height(Length::Units(1000));

        let graph_container = Container::new(graph)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(20)
            .center_x()
            .center_y()
            .into();

        let ram_maker = self.gpu_info.as_ref().and_then(|i| Some(i.ram_maker.to_string()));
        let memory_size = self.gpu_info.as_ref().and_then(|i| Some(i.memory.dedicated));
        let memory_size_value = memory_size.and_then(|s| Some( if s < Kibibytes(1024) {
            s.0 as f32
        } else {
            s.0 as f32 / 1024.0
        }));
        let memory_size_units = memory_size.and_then(|s| Some( if s < Kibibytes(1024) {
            "KiB"
        } else {
            "MiB"
        })).unwrap_or("");

        let ram_bus_width = self.gpu_info.as_ref().and_then(|i| Some(i.ram_bus_width));
        let core_clock = self.gpu_status.as_ref().and_then(|i| Some(i.clocks[&PublicClockId::Graphics].0 as f32 / 1000.0));
        let memory_clock = self.gpu_status.as_ref().and_then(|i| Some(i.clocks[&PublicClockId::Memory].0 as f32 / 1000.0));

        let test_settings_col = Column::with_children(vec![
            Row::with_children(vec![
                Self::mutable_number_cell(&mut self.lower_bound_state, "Lower Bound", self.bound_lower.0 / 1000, (self.gpu_list[self.gpu_idx as usize].mem_clock_limit_lower.0 / 1000, self.bound_upper.0 / 1000), "MHz", Message::UpdateLowerBound).into(),
                Self::mutable_number_cell(&mut self.upper_bound_state, "Upper Bound", self.bound_upper.0 / 1000, (self.bound_lower.0 / 1000, self.gpu_list[self.gpu_idx as usize].mem_clock_limit_upper.0 / 1000), "MHz", Message::UpdateUpperBound).into(),
            ])
                .spacing(3)
                .align_items(Align::Center)
                .into(),
            Self::mutable_number_cell(&mut self.initial_step_state, "Initial Step", self.initial_step.0 / 1000, (1, 100), "MHz", Message::UpdateInitialStep).into(),
            Button::<Message>::new(&mut self.test_button_state, Text::new("Test"))
                .on_press(Message::StartTest)
                .into(),
        ])
            .spacing(3)
            .align_items(Align::Center)
            .into();

        Column::with_children(vec![
            Self::option_text_cell(&mut self.memory_type_state, "Memory Type", ram_maker).into(),
            Self::option_number_cell(&mut self.memory_size_state, "Memory Capacity", memory_size_value, memory_size_units).into(),
            Self::option_number_cell(&mut self.bus_width_state, "Bus Width", ram_bus_width, "bits").into(),
            Self::option_number_cell(&mut self.core_clock_state, "Core Clock", core_clock, "MHz").into(),
            Self::option_number_cell(&mut self.memory_clock_state, "Memory Clock", memory_clock, "MHz").into(),
            Row::with_children(vec![graph_container, test_settings_col]).into(),
            footer.into(),
        ])
        .into()
    }
}

impl Program<Message> for &GraphCtx {
    fn draw(&self, bounds: Rectangle, _cursor: Cursor) -> Vec<Geometry> {
        let clock = self.cache.draw(bounds.size(), |mut frame| {
            let root = IcedBackend::new(&mut frame).unwrap().into_drawing_area();
            root.fill(&WHITE).unwrap();

            let root = root.margin(10, 10, 10, 10);
            // After this point, we should be able to draw construct a chart context
            let mut chart = ChartBuilder::on(&root)
                // Set the size of the label region
                .x_label_area_size(20)
                .y_label_area_size(40)
                // Finally attach a coordinate on the drawing area and make a chart context
                .build_cartesian_2d((self.x_low_bound.0 as f32 / 1000.0)..(self.x_high_bound.0 as f32 / 1000.0), 0f32..200f32)
                .unwrap();

            // Then we can draw a mesh
            chart
                .configure_mesh()
                // We can customize the maximum number of labels allowed for each axis
                .x_labels(5)
                .y_labels(5)
                // We can also change the format of the label text
                .y_label_formatter(&|x| format!("{:.3}", x))
                .draw()
                .unwrap();

            // And we can draw something in the drawing area
            chart
                .draw_series(LineSeries::new(
                    self.results.iter().map(|p| (p.offset.0 as f32 / 1000.0, p.write_bandwidth_gbps)),
                    &GREEN,
                ))
                .unwrap();

            chart
                .draw_series(LineSeries::new(
                    self.results.iter().map(|p| (p.offset.0 as f32 / 1000.0, p.copy_bandwidth_gbps)),
                    &YELLOW,
                ))
                .unwrap();

            chart
                .draw_series(LineSeries::new(
                    self.results.iter().map(|p| (p.offset.0 as f32 / 1000.0, p.read_bandwidth_gbps)),
                    &BLUE,
                ))
                .unwrap();

            chart
                .draw_series(LineSeries::new(
                    self.results.iter().map(|p| (p.offset.0 as f32 / 1000.0, p.read_errors as f32)),
                    &RED,
                ))
                .unwrap();
            // Similarly, we can draw point series
            // chart
            //     .draw_series(PointSeries::of_element(
            //         self.results.iter().map(|p| (p.offset.0 as f32 / 1000.0, p.copy_bandwidth_gbps)),
            //         5,
            //         &RED,
            //         &|c, s, st| {
            //             return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
            //                 + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
            //                 + PlottersText::new(format!("{:?}", c), (10, 0), ("sans-serif", 10).into_font());
            //         },
            //     ))
            //     .unwrap();
        });
        vec![clock]
    }
}
