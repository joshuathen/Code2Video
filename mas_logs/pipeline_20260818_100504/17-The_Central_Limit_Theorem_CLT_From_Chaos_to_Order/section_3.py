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
        lecture_lines = ["Take many samples of size n.", "Calculate the mean for each.", "Plot these means on a graph.", "Watch the Bell Curve emerge.", "Aggregation creates order from chaos."]
        self.setup_layout("The Experiment: The Magic of Aggregation", lecture_lines)
        
        # Assets
        marble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg")
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        # Helper for color
        def color_line(index, color):
            self.lecture[index].set_color(color)

        # === Animation for Lecture Line 1 ===
        color_line(0, "#FFFF33")
        samples = VGroup(*[marble.copy().scale(0.1) for _ in range(10)])
        self.place_in_area(samples, "A1", "C3")
        self.play(FadeIn(samples))

        # === Animation for Lecture Line 2 ===
        color_line(1, "#FF9933")
        mean_dots = VGroup(*[Dot(radius=0.05, color=BLUE) for _ in range(10)])
        self.place_in_area(mean_dots, "D1", "F6")
        self.play(FadeIn(mean_dots))

        # === Animation for Lecture Line 3 ===
        color_line(2, "#33CCFF")
        hist = VGroup(*[Rectangle(height=np.random.rand(), width=0.3, color=BLUE, fill_opacity=0.5) for _ in range(5)])
        hist.arrange(RIGHT, aligned_edge=DOWN)
        self.place_in_area(hist, "D1", "F6")
        self.play(Create(hist))

        # === Animation for Lecture Line 4 ===
        color_line(3, "#33FF57")
        bell = ball.copy().scale(2)
        self.place_in_area(bell, "A4", "F6")
        self.play(FadeOut(samples, mean_dots, hist), FadeIn(bell))

        # === Animation for Lecture Line 5 ===
        color_line(4, "#FF33FF")
        self.wait(1)
