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
        lecture_lines = [
            "Velocity is the derivative of position.",
            "Our robot moves at f of t equals t squared.",
            "At three seconds, the derivative is six.",
            "So velocity is six units per second.",
            "Calculus measures real-world motion precisely."
        ]
        self.setup_layout("Application: Real-time Velocity", lecture_lines)
        
        # Mobjects
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 10, 2], axis_config={"include_tip": True}).scale(0.5)
        graph = axes.plot(lambda t: t**2, color=BLUE)
        motion_curve = VGroup(axes, graph)
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").scale(0.3)
        point_indicator = Dot(color=YELLOW)
        slope_line = TangentLine(graph, alpha=0.75, length=1.5, color=RED)
        velocity_formula = MathTex(r"v(3) = 6", color=YELLOW)
        
        # Apply layout fixes from critiques
        self.place_in_area(motion_curve, 'D1', 'F6', scale_factor=0.7)
        self.place_at_grid(point_indicator, 'B4', scale_factor=0.6)
        self.place_in_area(velocity_formula, 'A4', 'C6', scale_factor=0.9)
        
        # Add robot
        robot.next_to(point_indicator, UP)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(motion_curve), FadeIn(robot))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.play(Create(point_indicator))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        self.play(Create(slope_line))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GREEN)
        self.play(Write(velocity_formula))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(PURPLE)
        self.wait(2)
