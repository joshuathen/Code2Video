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
        self.setup_layout("Applying the Theory: Rhythmic Patterns", [
            "Notes must fill the measure perfectly.",
            "Different durations add up to the total.",
            "Rhythm works like a musical puzzle."
        ])
        
        # Create notes
        row1 = VGroup(*[Circle(radius=0.3, color=WHITE, fill_opacity=0.5) for _ in range(4)])
        row2 = VGroup(*[Circle(radius=0.3, color=WHITE, fill_opacity=0.5) for _ in range(4)])
        
        # Position using the grid requirements
        self.place_at_grid(row1, 'A3', scale_factor=0.9)
        self.place_at_grid(row2, 'B3', scale_factor=0.9)
        
        # Load asset
        puzzle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/puzzle.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(row1), FadeIn(row2), self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        rhythmic_pattern_group = VGroup(row1, row2)
        self.place_in_area(rhythmic_pattern_group, 'A3', 'C6', scale_factor=0.8)
        self.play(
            self.lecture[1].animate.set_color("#FF00FF"),
            FadeIn(puzzle_icon.scale(0.5).next_to(rhythmic_pattern_group, RIGHT)),
            rhythmic_pattern_group.animate.set_color("#FF00FF")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_in_area(rhythmic_pattern_group, 'C2', 'E5', scale_factor=0.75)
        self.play(
            self.lecture[2].animate.set_color("#00FFFF"),
            row1[0].animate.set_color("#00FFFF"),
            row2[0].animate.set_color("#00FFFF")
        )
        self.wait(1)
