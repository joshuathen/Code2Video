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
        lecture_lines = [
            "Two thieves split a mixed bead necklace.",
            "K plus S cuts ensure fair division.",
            "Each thief receives identical bead types.",
            "Raccoons demonstrate equal distribution of resources.",
            "Fairness is guaranteed by simple cutting math."
        ]
        self.setup_layout("The Stolen Necklace Problem", lecture_lines)
        
        # Necklace and beads assets
        necklace = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/necklace.svg")
        beads_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beads.svg")
        
        # Fix 30: Adjust necklace position
        self.place_in_area(necklace, 'B2', 'B5', scale_factor=0.6)
        self.add(necklace)

        # Scale asset
        scale_obj = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        self.place_at_grid(scale_obj, 'F3', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#F39C12"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E74C3C"))
        
        # Fix 32: Use place_in_area for cuts
        cut_lines = VGroup(Line(UP*0.5, DOWN*0.5, color=RED), Line(UP*0.5, DOWN*0.5, color=RED))
        self.place_in_area(cut_lines, 'B2', 'D5', scale_factor=0.7)
        self.play(Create(cut_lines))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#2ECC71"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#3498DB"))
        # Fix 31: Labels with defined anchor points
        raccoon1 = Text("R1", color=BLUE)
        raccoon2 = Text("R2", color=YELLOW)
        self.place_at_grid(raccoon1, 'C2', scale_factor=0.5)
        self.place_at_grid(raccoon2, 'C5', scale_factor=0.5)
        self.play(Write(raccoon1), Write(raccoon2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.play(Indicate(scale_obj, color="#00FF00"))
        self.wait(1)
