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
        self.setup_layout("Conclusion: The Universal Language", [
            "Music is sound arranged in time.",
            "Measure theory is the underlying architecture.",
            "We now understand the math of music."
        ])
        
        # Define elements using SVG assets
        notation = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/notes.svg")
        notation.set_color(WHITE)
        
        grid_geo = VGroup(*[Square(side_length=0.4, color=WHITE) for _ in range(9)])
        grid_geo.arrange_in_grid(3, 3, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        # Optimized position: Avoiding clutter and overcrowding
        self.place_in_area(notation, 'C2', 'D4', scale_factor=0.7)
        self.add(notation)
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), run_time=1)
        self.play(FadeOut(notation), FadeIn(grid_geo), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        # Optimized position: Avoiding clutter and overcrowding
        self.place_in_area(grid_geo, 'C5', 'D6', scale_factor=0.9)
        grid_geo.set_color("#FFD700")
        self.play(self.lecture[1].animate.set_color("#FFD700"), run_time=1)
        
        # === Animation for Lecture Line 3 ===
        # Final fade out with lingering pulse asset
        final_pulse = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/notes.svg")
        final_pulse.set_color("#FFD700")
        self.place_at_grid(final_pulse, 'C3', scale_factor=1.5)
        
        self.play(self.lecture[2].animate.set_color("#FFD700"), run_time=1)
        self.play(FadeOut(grid_geo), FadeIn(final_pulse), run_time=1)
        self.play(final_pulse.animate.scale(1.5), FadeOut(final_pulse), run_time=1.5)
        self.wait(1)
