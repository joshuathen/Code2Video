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
        self.setup_layout("The Pi Emergence", [
            "Increase the second block's mass significantly.",
            "Collision counts yield pi's digits.",
            "Scaling masses extracts more digits.",
            "One hundred kilograms gives three collisions.",
            "Ten thousand kilograms yields thirty-one."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Increase the second block's mass significantly.
        self.lecture[0].set_color("#FF6347")
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        circle = Circle(radius=1.5, color="#FFFFFF")
        self.place_in_area(circle, 'A2', 'C4', scale_factor=0.6)
        self.play(Create(circle), FadeIn(block_icon.scale(0.5).next_to(circle, UP)))

        # === Animation for Lecture Line 2 ===
        # Collision counts yield pi's digits.
        self.lecture[1].set_color("#FFD700")
        circ = DashedVMobject(Circle(radius=1.5, color="#FFD700"), num_dashes=30)
        self.place_at_grid(circ, 'D2', scale_factor=0.7)
        self.play(Create(circ))

        # === Animation for Lecture Line 3 ===
        # Scaling masses extracts more digits.
        self.lecture[2].set_color("#00FFFF")
        ratio = MathTex(r"C/d = \pi", color="#00FFFF")
        block_icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        self.place_at_grid(ratio, 'D5', scale_factor=0.7)
        self.play(Write(ratio), FadeIn(block_icon2.scale(0.3).next_to(ratio, DOWN)))

        # === Animation for Lecture Line 4 ===
        # One hundred kilograms gives three collisions.
        self.lecture[3].set_color("#32CD32")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Ten thousand kilograms yields thirty-one.
        self.lecture[4].set_color("#FF69B4")
        self.wait(1)
