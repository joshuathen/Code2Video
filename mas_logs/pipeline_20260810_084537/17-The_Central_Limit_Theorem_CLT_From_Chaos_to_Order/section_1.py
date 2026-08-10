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
        lecture_lines = [
            "Distributions track frequency of occurrence.",
            "Normal distributions show bell curves.",
            "Other distributions appear skewed or uniform."
        ]
        self.setup_layout("Prerequisite: The Concept of Distributions", lecture_lines)
        
        # Initialize mobjects
        uniform_graph = Rectangle(width=2, height=1, color="#3498db").set_fill("#3498db", opacity=0.5)
        bell_curve = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-3, 3], color="#e67e22")
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg", color="#e67e22")
        skewed_graph = Polygon(np.array([-1, -1, 0]), np.array([1, -1, 0]), np.array([0, 1, 0]), color="#e74c3c").set_fill("#e74c3c", opacity=0.5)
        mean_label = Text("Mean", font_size=24, color=WHITE)
        animated_point = Dot(color=RED)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(uniform_graph, 'C2')
        self.play(FadeIn(uniform_graph))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#e67e22")
        self.place_in_area(bell_curve, 'C2', 'E5')
        self.place_at_grid(bell_icon, 'B5', scale_factor=0.5)
        self.play(FadeIn(bell_curve), FadeIn(bell_icon))
        
        # Fixes for critical issues
        self.place_at_grid(mean_label, 'C4', scale_factor=0.9)
        self.add(mean_label)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#e74c3c")
        self.place_at_grid(animated_point, 'B3', scale_factor=0.7)
        self.play(FadeIn(skewed_graph), FadeIn(animated_point))
        self.wait(1)
