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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout with the specific title and lecture lines
        self.setup_layout(
            "Prerequisite Knowledge: Conservation Laws", 
            [
                'Conservation of energy and momentum drive this system.', 
                'Energy creates an elliptical boundary for velocities.', 
                'These laws constrain every possible bounce.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color("#FFFF00")
        
        # Define formulas: replaced MathTex with Text to avoid LaTeX dependency
        ke_formula = Text("Kinetic Energy: ½mv²", color="#FFFF00")
        p_formula = Text("Momentum: mv", color="#FFFF00")
        formulas = VGroup(ke_formula, p_formula).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # Position formulas shifted right to avoid clutter (Issue #28)
        self.place_in_area(formulas, 'A4', 'B6', scale_factor=0.7)
        
        self.play(FadeIn(formulas))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight second line, dim the first
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # Coordinate system v1 and v2
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE},
            tips=True
        )
        # Using Text for labels to avoid MathTex/LaTeX calls
        v1_label = axes.get_x_axis_label(Text("v₁"), edge=RIGHT, direction=RIGHT).scale(0.8)
        v2_label = axes.get_y_axis_label(Text("v₂"), edge=UP, direction=UP).scale(0.8)
        
        # Blue ellipse representing energy conservation
        ellipse = Ellipse(width=3.5, height=2.0, color="#0000FF", stroke_width=4)
        
        graph_group = VGroup(axes, v1_label, v2_label, ellipse)
        # Position graph shifted right and scaled down (Issue #27)
        self.place_in_area(graph_group, 'C2', 'F6', scale_factor=0.8)
        
        self.play(Create(axes), Write(v1_label), Write(v2_label))
        self.play(Create(ellipse))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight third line, dim the second
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        # Show a dot on the ellipse
        start_point = ellipse.point_from_proportion(0.15)
        end_point = ellipse.point_from_proportion(0.65)
        
        dot = Dot(point=start_point, color=WHITE, radius=0.08)
        
        # Visualizing a momentum jump (a chord)
        chord_line = Line(start_point, end_point, color=WHITE, stroke_opacity=0.3)
        
        self.play(FadeIn(dot))
        self.wait(1)
        
        # Animate the dot moving along the chord (straight line)
        self.play(Create(chord_line), run_time=0.5)
        self.play(dot.animate.move_to(end_point), run_time=2, rate_func=linear)
        self.wait(3)
