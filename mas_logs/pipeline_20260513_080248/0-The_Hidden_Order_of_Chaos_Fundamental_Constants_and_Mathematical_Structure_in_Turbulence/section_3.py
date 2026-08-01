from manim import *
import numpy as np

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
        # Setup the scene with specific title and lines
        lines = [
            'Kinetic energy flows from large swirls to smaller eddies.', 
            'Richardson described this process as an energy cascade.', 
            'Large whorls break down, feeding velocity into smaller ones.', 
            'Eventually, tiny scales turn kinetic energy into heat.', 
            'Viscosity acts as the final sink for turbulent motion.'
        ]
        self.setup_layout("The Energy Cascade: Richardson’s Poem", lines)

        # === Animation for Lecture Line 1 ===
        # Large, thick blue circle rotating slowly
        self.lecture[0].set_color("#00008B")
        large_eddy = Circle(radius=2.0, color="#00008B", stroke_width=10)
        self.place_in_area(large_eddy, "A2", "D5")
        
        # Add a swirl arrow for visual motion cues
        swirl_arrow = Arc(radius=1.5, start_angle=0, angle=PI, color="#00008B").add_tip()
        swirl_arrow.move_to(large_eddy.get_center())
        
        self.play(Create(large_eddy), Create(swirl_arrow), run_time=1)
        self.play(Rotate(large_eddy, angle=PI, run_time=2), Rotate(swirl_arrow, angle=PI, run_time=2))

        # === Animation for Lecture Line 2 ===
        # Large circle splits into four medium-sized blue circles rotating faster
        self.lecture[1].set_color("#4169E1")
        
        medium_eddies = VGroup(*[
            Circle(radius=0.8, color="#4169E1", stroke_width=6) for _ in range(4)
        ])
        
        # Distribute medium eddies across quadrants within A2 to D5 as requested
        self.place_in_area(medium_eddies[0], "A2", "B3")
        self.place_in_area(medium_eddies[1], "A4", "B5")
        self.place_in_area(medium_eddies[2], "C2", "D3")
        self.place_in_area(medium_eddies[3], "C4", "D5")

        self.play(
            FadeOut(swirl_arrow),
            ReplacementTransform(large_eddy, medium_eddies),
            run_time=1.5
        )
        self.play(*[Rotate(m, angle=2*PI, run_time=2) for m in medium_eddies])

        # === Animation for Lecture Line 3 ===
        # Medium circles divide into four small blue circles each (16 total)
        self.lecture[2].set_color("#87CEEB")
        
        small_eddies = VGroup()
        for i in range(4):
            parent_center = medium_eddies[i].get_center()
            # Position small circles in a cluster around the parent medium eddy center
            for dx, dy in [(-0.35, 0.35), (0.35, 0.35), (-0.35, -0.35), (0.35, -0.35)]:
                s_eddy = Circle(radius=0.25, color="#87CEEB", stroke_width=3)
                s_eddy.move_to(parent_center + np.array([dx, dy, 0]))
                small_eddies.add(s_eddy)

        self.play(ReplacementTransform(medium_eddies, small_eddies), run_time=1.5)
        self.play(*[Rotate(s, angle=3*PI, run_time=2) for s in small_eddies])

        # === Animation for Lecture Line 4 ===
        # Smallest circles dissipate into tiny red dots representing heat
        self.lecture[3].set_color("#FF4500")
        
        heat_dots = VGroup(*[
            Dot(point=s.get_center(), color="#FF4500", radius=0.06) for s in small_eddies
        ])
        
        self.play(ReplacementTransform(small_eddies, heat_dots), run_time=1.5)

        # === Animation for Lecture Line 5 ===
        # Highlight VISCOSITY with white pulse effect as dots disappear
        self.lecture[4].set_color("#FFFFFF")
        
        viscosity_label = Text("VISCOSITY", font_size=36, color=WHITE)
        self.place_in_area(viscosity_label, "F2", "F5") # Positioned to prevent clipping
        
        self.play(FadeIn(viscosity_label))
        
        # Pulse effect and kinetic energy dots dissipation
        self.play(
            viscosity_label.animate.scale(1.2),
            heat_dots.animate.set_opacity(0).scale(0.5),
            run_time=0.5
        )
        self.play(
            viscosity_label.animate.scale(1/1.2),
            run_time=0.5
        )
        
        self.wait(2)
