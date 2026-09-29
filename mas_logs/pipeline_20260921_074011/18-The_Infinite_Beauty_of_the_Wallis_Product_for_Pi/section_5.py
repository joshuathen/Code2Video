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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Wallis product converges slowly to Pi.", "Modern series are much more efficient.", "It is a beautiful, but steady race."]
        self.setup_layout("Application: The Convergence Race", lecture_lines)
        
        # Assets
        stopwatch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stopwatch.svg")
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#DC143C")
        tortoise = Text("🐢", font_size=48)
        self.place_at_grid(tortoise, "B5", scale_factor=0.6)
        bar1 = Rectangle(width=4, height=0.3, color="#DC143C", fill_opacity=0.5)
        self.place_in_area(bar1, "C4", "C6", scale_factor=0.7)
        self.place_at_grid(stopwatch, "B2", scale_factor=0.3)
        self.play(FadeIn(tortoise), Create(bar1), FadeIn(stopwatch))
        self.play(tortoise.animate.shift(RIGHT * 1), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#ADFF2F")
        hare = Text("🐇", font_size=48)
        self.place_at_grid(hare, "D5", scale_factor=0.6)
        bar2 = Rectangle(width=4, height=0.3, color="#ADFF2F", fill_opacity=0.5)
        self.place_in_area(bar2, "E4", "E6", scale_factor=0.7)
        self.play(FadeIn(hare), Create(bar2))
        self.play(hare.animate.shift(RIGHT * 3), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        pi_val = MathTex(r"\\pi \\approx 3.14159...", color="#FFD700")
        self.place_at_grid(pi_val, "F5", scale_factor=0.9)
        self.place_at_grid(magnifying_glass, "F2", scale_factor=0.5)
        self.play(Write(pi_val), FadeIn(magnifying_glass))
        self.play(Indicate(pi_val), run_time=2)
        self.wait(1)
