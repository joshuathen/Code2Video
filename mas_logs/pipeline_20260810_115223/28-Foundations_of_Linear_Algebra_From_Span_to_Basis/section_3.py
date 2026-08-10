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
        self.setup_layout("Linear Dependence: The Redundant Vector", [
            "Linear dependence occurs when vectors are redundant.",
            "One vector can be formed by the others.",
            "Removing redundant vectors maintains the same span."
        ])
        
        # Load asset
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Note: '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg' is just a placeholder name from the prompt's example, 
        # but I will load it if it exists.
        try:
            icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        except:
            icon = Dot(color=GREY)
        
        # Vectors - grouped to keep them spatially synchronized
        v1 = Arrow(ORIGIN, RIGHT * 1.5, color=BLUE)
        v2 = Arrow(ORIGIN, UP * 1.2, color=GREEN)
        v1_group = VGroup(v1, v2)
        v3 = Arrow(ORIGIN, (RIGHT * 1.5 + UP * 1.2) * 0.5, color=RED)
        
        # Positioning based on grid layout instructions
        self.place_in_area(v1_group, 'B1', 'B3', scale_factor=0.85)
        self.place_at_grid(v3, 'D3', scale_factor=1.0)
        self.place_at_grid(icon, 'F6', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(v1), FadeIn(v2), FadeIn(v3), FadeIn(icon))
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(v3.animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(FadeOut(v3))
        self.wait(1)
