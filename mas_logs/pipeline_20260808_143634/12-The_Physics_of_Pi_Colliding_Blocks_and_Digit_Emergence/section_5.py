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
        lecture_lines = [
            "Collisions track angle in phase space.",
            "Pi emerges from geometric physical constraints.",
            "Blocks act as a physical calculation engine."
        ]
        self.setup_layout("Conclusion and Intuition", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        summary_text = Text("Phase Space: Angle = Arc Length", font_size=28).set_color("#FFFFFF")
        self.place_in_area(summary_text, 'A4', 'B6', scale_factor=0.9)
        self.play(Write(summary_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        constraint_text = Text("Physical Constraints: Energy Conservation", font_size=28).set_color("#FFFFFF")
        self.place_in_area(constraint_text, 'C4', 'D6', scale_factor=0.8)
        self.play(Write(constraint_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        # Add Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        self.place_at_grid(blocks, 'E4', scale_factor=1.0)
        box = SurroundingRectangle(blocks, color="#FFC300", buff=0.2)
        
        self.play(FadeIn(blocks), Create(box))
        self.wait(2)
        
        self.play(FadeOut(*self.mobjects))
        self.wait(1)
