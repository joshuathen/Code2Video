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
        self.setup_layout("Key Conditions for Validity", [
            "Samples must be independent and random.",
            "Use a sample size of thirty or more.",
            "These conditions guarantee a normal curve."
        ])
        
        # Assets
        dice = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # === Animation for Lecture Line 1 ===
        indep_text = Text("Independent Trials", color="#00FF00")
        self.place_at_grid(indep_text, "B3", scale_factor=0.9 * 0.75)
        self.place_at_grid(dice, "B2", scale_factor=0.5)
        dice.next_to(indep_text, LEFT, buff=0.2)
        
        self.play(FadeIn(indep_text), FadeIn(dice))
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 2 ===
        large_sample = Text("Large Sample Size", color="#00FF00")
        self.place_at_grid(large_sample, "C3", scale_factor=0.9 * 0.75)
        self.place_at_grid(ruler, "C2", scale_factor=0.5)
        ruler.next_to(large_sample, LEFT, buff=0.2)
        
        self.play(FadeIn(large_sample), FadeIn(ruler))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 3 ===
        check1 = Tex(r"$\checkmark$", color="#FFFF00").scale(1.5)
        self.place_at_grid(check1, "B4", scale_factor=0.8)
        
        check2 = Tex(r"$\checkmark$", color="#FFFF00").scale(1.5)
        self.place_at_grid(check2, "C4", scale_factor=0.8)
        
        self.play(Create(check1), Create(check2))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
