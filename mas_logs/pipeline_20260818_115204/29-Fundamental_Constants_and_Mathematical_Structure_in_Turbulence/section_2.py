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
        self.setup_layout("The Kolmogorov Cascade: Energy Transfer", [
            "Large eddies break down into smaller energy-containing ones.",
            "Energy cascades down to the smallest viscous scales.",
            "The energy spectrum follows a minus five-thirds law."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Draw large circles representing eddies that split into smaller ones
        self.lecture[0].set_color(YELLOW)
        
        eddy_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/whirlpool.svg"
        large_eddy = SVGMobject(eddy_asset).set_color(BLUE)
        self.place_at_grid(large_eddy, 'B3', scale_factor=1.5)
        self.play(FadeIn(large_eddy))
        
        small_eddies = VGroup(*[SVGMobject(eddy_asset).set_color(TEAL) for _ in range(4)])
        self.place_at_grid(small_eddies[0], 'D2', scale_factor=0.6)
        self.place_at_grid(small_eddies[1], 'D3', scale_factor=0.6)
        self.place_at_grid(small_eddies[2], 'D4', scale_factor=0.6)
        self.place_at_grid(small_eddies[3], 'D5', scale_factor=0.6)
        
        self.play(TransformFromCopy(large_eddy, small_eddies), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        energy_arrows = VGroup(*[Arrow(start=large_eddy.get_center(), end=e.get_center(), buff=0.1, color=ORANGE) for e in small_eddies])
        self.play(Create(energy_arrows))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        axes = Axes(x_range=[0, 6], y_range=[0, 4], axis_config={"include_tip": False}).scale(0.6)
        self.place_at_grid(axes, 'F3', scale_factor=0.8)
        graph = axes.plot(lambda x: x**(-5/3) * 5, x_range=[1, 5], color=RED)
        
        self.play(Create(axes), Create(graph))
        self.wait(2)
