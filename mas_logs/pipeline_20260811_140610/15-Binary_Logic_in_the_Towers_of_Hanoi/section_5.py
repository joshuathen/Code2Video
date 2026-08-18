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
        self.setup_layout("Summary & Application", [
            "Recursive problems map to binary.",
            "Simple patterns solve complex puzzles.",
            "Binary is the foundation of logic."
        ])
        
        # Asset Loading
        puzzle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/puzzle.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#87CEEB")
        
        # Visual: Binary-Tower mapping + puzzle icon
        mapping_text = Text("Binary & Recursion overlap", color="#87CEEB", font_size=24)
        group = VGroup(mapping_text, puzzle_icon).arrange(DOWN)
        
        # Apply layout fixes from Issues 34 & 35: Use columns 4-6
        self.place_in_area(group, "A4", "F6", scale_factor=0.6)
        
        self.play(FadeIn(group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Visual: 2^n - 1 formula
        formula = MathTex("2^n - 1", color="#FFD700")
        
        # Apply layout fix from Issue 33
        self.place_at_grid(formula, "E2", scale_factor=1.2)
        
        self.play(FadeIn(formula), Flash(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Final cleanup
        self.play(FadeOut(group), FadeOut(formula))
        self.play(FadeOut(self.lecture), FadeOut(self.title))
        self.wait(1)
