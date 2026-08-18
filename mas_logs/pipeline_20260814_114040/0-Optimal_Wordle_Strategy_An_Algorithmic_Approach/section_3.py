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
        self.setup_layout("The Strategy: Frequency vs. Positional Probability", [
            "Letter frequency determines word strength.", 
            "We map frequency across all slots.", 
            "This calculates a word's weighted score."
        ])
        
        # --- Preparation ---
        freq_text = Text("Frequency: E=12%, Z=0.1%", color="#FFFFFF")
        # Use SVGMobject as requested in storyboards
        pos_grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg", color="#00FFFF")
        score_calc = MathTex(r"Score = \sum P(char, slot)", color="#FF00FF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        # Fix 27/42: freq_text position
        self.place_in_area(freq_text, 'B2', 'B5', scale_factor=0.9)
        self.play(Write(freq_text))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Fix 28/43: pos_grid position
        self.place_in_area(pos_grid, 'C3', 'D6', scale_factor=0.9)
        self.play(FadeIn(pos_grid))
        self.play(freq_text.animate.set_color("#00FFFF"))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        # Fix 29/44: score_calc position
        self.place_at_grid(score_calc, 'E4', scale_factor=0.8)
        self.play(Write(score_calc))
        self.play(Indicate(pos_grid))
        self.wait(2)
