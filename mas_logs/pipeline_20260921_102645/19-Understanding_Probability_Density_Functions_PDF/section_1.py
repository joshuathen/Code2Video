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
        self.setup_layout("Prerequisite: From Discrete to Continuous", [
            "Discrete events use histograms with bars.",
            "Continuous data makes bar width zero.",
            "Density functions bridge this transition."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Using SVG Asset
        hist_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg")
        bars = VGroup(*[
            Rectangle(width=0.6, height=h, color=BLUE_B, fill_opacity=0.7)
            for h in [1.5, 2.5, 2.0, 1.0]
        ]).arrange(RIGHT, aligned_edge=DOWN, buff=0.1)
        
        # Integrate icon with bars
        hist_group = VGroup(bars, hist_icon).arrange(DOWN)
        
        self.place_at_grid(hist_group, 'D2', scale_factor=0.8)
        self.play(FadeIn(hist_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        thin_bars = VGroup(*[
            Rectangle(width=0.2, height=h, color=BLUE_B, fill_opacity=0.7)
            for h in [0.5, 1.2, 2.0, 3.0, 2.5, 1.8, 0.8]
        ]).arrange(RIGHT, aligned_edge=DOWN, buff=0.05)
        
        self.place_at_grid(thin_bars, 'D2', scale_factor=0.8)
        self.play(Transform(bars, thin_bars))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        curve = FunctionGraph(lambda x: 3 * np.exp(-x**2), x_range=[-2, 2], color=RED)
        self.place_at_grid(curve, 'C4', scale_factor=0.8)
        
        # Adding asset to the final state as requested
        final_hist_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg")
        self.place_at_grid(final_hist_icon, 'E4', scale_factor=0.3)
        final_hist_icon.set_color("#00FFFF") # Color #00FFFF
        
        self.play(FadeOut(bars), Create(curve), FadeIn(final_hist_icon))
        area = ImplicitFunction(lambda x, y: (y <= 3 * np.exp(-x**2)).astype(float) - (y >= 0).astype(float), color="#00FFFF", fill_opacity=0.3).scale(0.8).move_to(curve.get_center())
        self.play(FadeIn(area))
        self.wait(2)
