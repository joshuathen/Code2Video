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
        self.setup_layout("Building the Sequence: Data Collection", [
            "Count regions for n points.", 
            "The sequence: 1, 2, 4, 8, 16.", 
            "Pattern suggests two to power n-1."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        circle = Circle(radius=1.2, color=WHITE)
        # Applying fix for Issue 23 and 25
        self.place_in_area(circle, 'B4', 'E6', scale_factor=0.7)
        self.play(Create(circle))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF4500")
        
        # n=3
        pt3 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg")
        self.place_at_grid(pt3, 'B4', scale_factor=0.5)
        self.play(FadeIn(pt3))
        
        # n=4
        pt4 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg")
        self.place_at_grid(pt4, 'C4', scale_factor=0.5)
        self.play(FadeIn(pt4))
        
        # n=5
        pt5 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg")
        self.place_at_grid(pt5, 'D4', scale_factor=0.5)
        self.play(FadeIn(pt5))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#1E90FF")
        # Applying fix for Issue 24
        hypo_text = Text("Hypothesis: 2^(n-1)", font_size=20, color="#FF00FF")
        self.place_at_grid(hypo_text, 'E4', scale_factor=0.8)
        self.play(Write(hypo_text))
        self.wait(2)
