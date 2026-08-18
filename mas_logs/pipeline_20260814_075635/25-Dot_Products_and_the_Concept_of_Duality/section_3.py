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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Linear functionals are essentially dot products.", "Use contour lines to represent the function.", "Sensors detect values at specific coordinates."]
        self.setup_layout("Visualizing Linear Functionals", lecture_lines)
        
        # Grid axes
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_numbers": False}).scale(0.5)
        self.place_in_area(axes, 'D2', 'F6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes))
        self.lecture[0].set_color(YELLOW)
        
        # === Animation for Lecture Line 2 ===
        # Parallel lines as contour lines
        lines = VGroup(*[Line(start=[-2, i, 0], end=[2, i, 0], color=BLUE) for i in np.linspace(-2, 2, 9)])
        lines.apply_matrix(np.array([[1, 0.5], [0, 1]])) # Shear
        self.place_in_area(lines, 'D2', 'F6', scale_factor=0.6)
        self.play(Create(lines))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # === Animation for Lecture Line 3 ===
        # Representing a sensor/point
        sensor_point = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg", color=RED)
        sensor_label = Text("Sensor", font_size=16, color=RED)
        
        self.place_at_grid(sensor_label, 'D4', scale_factor=0.7)
        self.place_at_grid(sensor_point, 'D5', scale_factor=0.7)
        
        self.play(FadeIn(sensor_point), Write(sensor_label))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
