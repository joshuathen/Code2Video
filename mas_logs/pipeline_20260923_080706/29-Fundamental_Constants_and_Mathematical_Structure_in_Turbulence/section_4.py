from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Role of Reynolds Number as a Scaling Constant", [
            "Reynolds number dictates turbulence intensity.", 
            "Higher Re increases the range of scales.", 
            "Scale separation grows with Reynolds number."
        ])
        
        # Assets
        pipe_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pipe.svg")
        wing_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wing.svg")
        
        # Animation Elements
        re_label = MathTex("Re", font_size=48, color=BLUE)
        self.place_at_grid(re_label, 'B4', scale_factor=0.8)
        
        # Slider simulation
        slider_line = Line(start=self.grid["D2"], end=self.grid["D6"], color=WHITE)
        slider_knob = Dot(self.grid["D2"], color=RED)
        
        # Visualizing turbulence intensity
        flow_indicator = Circle(radius=0.5, color=WHITE)
        self.place_at_grid(flow_indicator, 'C4', scale_factor=0.7)

        # Asset group
        visual_group = VGroup(pipe_icon, wing_icon)
        self.place_in_area(visual_group, 'A4', 'A6', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(re_label), Write(self.lecture[0]))
        self.lecture[0].set_color(BLUE)
        self.play(Create(slider_line), FadeIn(slider_knob), FadeIn(pipe_icon))

        # === Animation for Lecture Line 2 ===
        self.play(
            slider_knob.animate.move_to(self.grid["D6"]),
            Write(self.lecture[1]),
            run_time=2
        )
        self.lecture[1].set_color(YELLOW)
        
        # Change flow
        turbulent_flow = VGroup(*[Circle(radius=0.1, color=WHITE).shift(np.random.rand(3)*0.5) for _ in range(10)])
        self.play(Transform(flow_indicator, turbulent_flow), FadeIn(wing_icon))

        # === Animation for Lecture Line 3 ===
        self.play(Write(self.lecture[2]))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
