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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Derivatives find the slope of a curve.",
            "Integrals find the area under a curve.",
            "These are inverse mathematical operations."
        ]
        self.setup_layout("Visualizing the Inverse Operation", lecture_lines)
        
        # Colors for lecture lines
        c1 = "#FF9AA2"
        c2 = "#B5EAD7"
        c3 = "#C7CEEA"
        
        # Assets
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        
        # Axes
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * x**2, color=WHITE)
        graph_group = VGroup(axes, curve)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(c1)
        self.place_in_area(axes, 'A3', 'D6', scale_factor=0.5)
        
        tangent_line = Line(start=LEFT, end=RIGHT, color=c1).scale(0.5)
        self.place_at_grid(tangent_line, 'C4', scale_factor=0.6)
        
        self.place_at_grid(speedometer, 'B5', scale_factor=0.5)
        self.play(FadeIn(axes), FadeIn(curve), FadeIn(speedometer), FadeIn(tangent_line))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(c2)
        area = axes.get_area(curve, x_range=[0, 3], color=c2, opacity=0.5)
        self.play(FadeIn(area))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(c3)
        self.place_at_grid(calculator, 'E5', scale_factor=0.5)
        self.play(FadeIn(calculator))
        self.play(Indicate(curve, color=c3))
        self.wait(1)
