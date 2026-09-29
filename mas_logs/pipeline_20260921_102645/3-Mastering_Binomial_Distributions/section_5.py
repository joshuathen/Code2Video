from manim import *
import os

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
        self.setup_layout("Application & Wrap-up", [
            "Binomial distributions model real-world fixed trials.",
            "Useful for quality control and genetic predictions.",
            "Essential for predicting experiment outcomes accurately."
        ])
        
        # Define visual elements with assets
        coin_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg"
        die_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/die.svg"
        
        coin = SVGMobject(coin_path) if os.path.exists(coin_path) else Circle(radius=0.3, color=YELLOW)
        die = SVGMobject(die_path) if os.path.exists(die_path) else Square(side_length=0.5, color=WHITE)
        qc_icon = VGroup(Square(side_length=0.5, color=BLUE), Text("QC", font_size=12)).arrange(DOWN)
        dna_icon = VGroup(Line(ORIGIN, UP*0.5, color=GREEN), Line(UP*0.5, UP, color=GREEN)).rotate(PI/4)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(die, 'B2', scale_factor=0.8)
        self.play(FadeIn(die))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.place_at_grid(qc_icon, 'B4', scale_factor=0.9)
        self.place_at_grid(dna_icon, 'B6', scale_factor=0.9)
        self.play(FadeIn(qc_icon), FadeIn(dna_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        final_text = Text("Predict Outcome = Success!", font_size=24, color=WHITE)
        self.place_in_area(final_text, 'D2', 'E6', scale_factor=0.75)
        self.play(Write(final_text))
        self.wait(2)
