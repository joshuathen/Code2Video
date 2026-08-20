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
        self.setup_layout("The Math: Chain Rule in Motion", [
            "The chain rule connects small weight changes.",
            "It acts as a precise mathematical multiplier.",
            "Tiny changes ripple through connected network gears."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Use gear asset for chain rule visualization
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg
        gear_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg"
        
        # Load and configure gears
        gears = VGroup(*[SVGMobject(gear_path, color="#A9A9A9") for _ in range(3)])
        gears.arrange(RIGHT, buff=0.2)
        
        # Place gears in area C2-E5 per feedback
        self.place_in_area(gears, "C2", "E5", scale_factor=1.0)
        
        # Label gear rotation with white tick marks
        ticks = VGroup(*[Line(UP*0.1, DOWN*0.1, color=WHITE).next_to(gears[i], UP, buff=0.1) for i in range(3)])
        
        self.play(FadeIn(gears), FadeIn(ticks))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Animate gear rotation
        self.play(*[Rotate(gears[i], angle=2*PI, run_time=1.5) for i in range(3)])
        self.lecture[1].set_color("#D3D3D3")

        # === Animation for Lecture Line 3 ===
        # Show ripples of light in cyan (#00FFFF)
        ripples = VGroup(*[Circle(radius=0.3, color="#00FFFF", stroke_width=3).move_to(gears[i].get_center()) for i in range(3)])
        
        self.play(
            *[Flash(gears[i].get_center(), color="#00FFFF", line_length=0.2, num_lines=8) for i in range(3)],
            run_time=2.0
        )
            
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
