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
        self.setup_layout("Real-World Application: The Saccharimeter", [
            "Saccharimeters measure sugar purity in industry.",
            "They observe rotation to find concentration.",
            "Precision ensures recipes match desired sweetness."
        ])
        
        # --- Visual Elements ---
        # Schematic image
        schematic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sample.svg", color=WHITE)
        
        # Light source/laser
        laser = Line(start=ORIGIN, end=RIGHT*0.5, color=WHITE).shift(LEFT*0.5)
        laser_box = Rectangle(height=0.4, width=0.4, color=WHITE).next_to(laser, LEFT, buff=0)
        
        # Sample cell
        cell = Rectangle(height=0.8, width=2.0, color=WHITE)
        
        # Analyzer/Detector
        analyzer = Rectangle(height=0.6, width=0.3, color=WHITE)
        
        # Polarized light beam
        beam = Line(start=LEFT*1.0, end=RIGHT*1.0, color=WHITE)
        
        # Layout on grid
        self.place_at_grid(schematic, 'B2', scale_factor=0.5)
        self.place_at_grid(VGroup(laser_box, laser), 'D1', scale_factor=0.7)
        self.place_at_grid(cell, 'D2', scale_factor=0.8)
        self.place_at_grid(analyzer, 'D5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(schematic), Create(laser_box), Create(laser), Create(cell), Create(analyzer))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Beam passing and rotating
        beam.set_stroke(width=4)
        self.place_in_area(beam, 'D3', 'D4', scale_factor=1.0)
        self.play(Create(beam))
        self.play(Rotate(beam, angle=PI/6, about_point=self.grid['D3']))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFCC00"))
        # Adjust analyzer
        self.play(Rotate(analyzer, angle=PI/6, about_point=self.grid['D5']))
        self.wait(1)
