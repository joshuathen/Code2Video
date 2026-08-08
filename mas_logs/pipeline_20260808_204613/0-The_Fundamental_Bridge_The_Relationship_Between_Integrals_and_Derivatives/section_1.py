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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Intuitive Hook: The Velocity-Position Analogy", [
            "A robot moves along a straight path.",
            "Speedometer shows instantaneous velocity at each moment.",
            "Distance is the accumulated velocity over time.",
            "Area under speed curve equals total distance.",
            "Integration and differentiation link these concepts."
        ])
        
        # Mobjects
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color="#FFD700")
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        path = Line(start=self.grid["D2"], end=self.grid["D6"], color=WHITE)
        speed_graph = Axes(
            x_range=[0, 4, 1], y_range=[0, 3, 1],
            axis_config={"include_numbers": False}
        ).scale(0.3)
        velocity_label = MathTex("v(t)", color="#FFD700").scale(0.8)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_at_grid(path, "D4")
        self.place_at_grid(robot, "D3", scale_factor=1.0)
        self.play(Create(path), FadeIn(robot))
        self.play(robot.animate.move_to(self.grid["D6"]), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.place_in_area(speed_graph, "B3", "C6", scale_factor=0.9)
        self.place_at_grid(velocity_label, "B2", scale_factor=0.8)
        self.place_at_grid(speedometer, "A5", scale_factor=0.5)
        self.play(Create(speed_graph), FadeIn(velocity_label), FadeIn(speedometer))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00BFFF")
        tracer = Dot(color="#00BFFF", radius=0.1)
        self.place_at_grid(tracer, "D3", scale_factor=1.0)
        self.play(FadeIn(tracer))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00BFFF")
        area = speed_graph.get_area(speed_graph.plot(lambda x: 2), color="#00BFFF", opacity=0.5)
        self.play(Create(area))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        self.wait(1)
