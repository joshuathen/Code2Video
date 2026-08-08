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
        self.setup_layout("The Chain Rule: The Domino Effect", [
            "The chain rule explains dependent changes.", 
            "Error output traces back to weights.", 
            "Think of it as falling dominos."
        ])
        
        # Create dominoes using asset
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/domino.svg"
        dominos = VGroup(*[SVGMobject(asset_path, color=WHITE) for _ in range(5)])
        dominos.arrange(RIGHT, buff=0.3)
        self.place_in_area(dominos, "B5", "E6", scale_factor=0.35)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(dominos))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # First domino falls
        first_domino = dominos[0]
        self.play(Rotate(first_domino, angle=-PI/2, about_point=first_domino.get_bottom()), 
                  self.lecture[1].animate.set_color("#FF0000"))
        
        # Sequential fall
        for i in range(1, 5):
            self.play(Rotate(dominos[i], angle=-PI/2, about_point=dominos[i].get_bottom()), run_time=0.2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Flash sequence travels from last to first
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        for i in range(4, -1, -1):
            self.play(dominos[i].animate.set_color("#FFFF00"), run_time=0.2)
        self.wait(2)
