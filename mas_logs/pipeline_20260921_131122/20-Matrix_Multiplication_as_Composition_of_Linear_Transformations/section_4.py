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
        self.setup_layout("Visualizing Non-Commutativity", [
            "Order of operations matters significantly.", 
            "Apply A then B differs.", 
            "Applying B then A yields another."
        ])
        
        # Objects to animate
        # Use SVG assets as requested
        obj_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color="#D3D3D3")
        obj_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color="#A9A9A9")
        
        # Applied visual feedback fixes:
        # Align with text flow vertically (using B3 and B4 based on B3/B4 suggestion)
        # Using scale 0.7 for better visibility
        self.place_at_grid(obj_a, 'B3', scale_factor=0.7)
        self.place_at_grid(obj_b, 'B4', scale_factor=0.7)
        self.add(obj_a, obj_b)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFE0")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFDAB9")
        # Apply A then B: A=Rotate(90), B=Shift(right)
        self.play(
            Rotate(obj_a, angle=PI/2),
            obj_a.animate.shift(RIGHT * 1),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#E0FFFF")
        # Applying B then A: B=Shift(right), A=Rotate(90)
        self.play(
            obj_b.animate.shift(RIGHT * 1),
            Rotate(obj_b, angle=PI/2),
            run_time=1
        )
        # Final objects, including chair.svg for mismatch visual
        chair = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chair.svg")
        self.place_at_grid(chair, 'E3', scale_factor=0.5)
        self.play(FadeIn(chair))
        self.wait(2)
