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
        self.setup_layout("Robot Battery Example", [
            "We model battery life with a PDF.",
            "Find probability by shading an interval.",
            "Calculate area between four and six."
        ])
        
        # Add assets
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot_icon, "A6", scale_factor=0.3)
        self.add(robot_icon)

        # Define the PDF (simple triangle/trapezoid for demo)
        axes = Axes(x_range=[0, 10, 2], y_range=[0, 0.5, 0.1], axis_config={"include_numbers": True}).scale(0.5)
        self.place_in_area(axes, 'A1', 'C6', scale_factor=0.7)
        self.add(axes)
        
        battery_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/battery.svg")
        self.place_at_grid(battery_icon, "A1", scale_factor=0.2)
        self.add(battery_icon)
        
        pdf = axes.plot(lambda x: 0.05 * x if x < 5 else 0.05 * (10 - x), x_range=[0, 10], color=BLUE)
        self.add(pdf)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        
        # Shade region [4, 6]
        shaded_area = axes.get_area(pdf, x_range=[4, 6], color=RED, opacity=0.5)
        self.play(FadeIn(shaded_area))
        
        label = Text("Probability of failure", font_size=18, color=WHITE)
        self.place_at_grid(label, 'D3', scale_factor=0.8)
        self.play(Write(label))
        
        integral = MathTex(r"\\int_{4}^{6} f(x) \\, dx", font_size=24).set_color(YELLOW)
        self.place_at_grid(integral, 'E3', scale_factor=1.0)
        self.play(Write(integral))
        self.wait(2)
