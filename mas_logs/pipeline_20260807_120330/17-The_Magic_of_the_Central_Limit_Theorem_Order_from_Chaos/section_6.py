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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with lecture lines from storyboard
        self.setup_layout("Application: Why Does This Matter?", [
            "The CLT allows confident predictions about unknown populations.",
            "It's the secret foundation for polling and scientific research.",
            "Order emerges from chaos, making the unpredictable predictable."
        ])
        
        # Colors (aligned with lines)
        color_line1 = "#FFFF66"  # Light Yellow
        color_line2 = "#66CCFF"  # Light Blue
        color_line3 = "#FF6666"  # Light Red
        
        # === Animation for Lecture Line 1 ===
        # Use color_line1 for the lecture line and its associated visual elements
        self.lecture[0].set_color(color_line1)
        
        # Elevator visual: A frame representing the physical constraint
        elevator_frame = Rectangle(width=3, height=4, color=color_line1)
        self.place_in_area(elevator_frame, 'B2', 'E4')
        
        # Max Weight line at the top of the elevator
        max_y = self.grid['B2'][1] - 0.2
        max_weight_line = Line(
            start=[self.grid['B2'][0], max_y, 0],
            end=[self.grid['B4'][0], max_y, 0],
            color=color_line1
        )
        
        # [Issue 37 Fix]: Position 'Max Weight' label at B6 to avoid crowding
        max_weight_label = Text("Max Weight", font_size=18, color=color_line1)
        self.place_at_grid(max_weight_label, 'B6', scale_factor=0.8)

        self.play(Create(elevator_frame))
        self.play(Create(max_weight_line), Write(max_weight_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Reset previous line color and highlight current line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_line2)
        
        # Bell curve representing the CLT's prediction of sum weights
        mu = self.grid['D3'][1] # Centered near the bottom of the elevator
        sigma = 0.5
        
        def vertical_pdf_func(y):
            # Normal distribution function oriented vertically
            # Peak at mu, base at x=1.5
            return self.grid['C2'][0] + 1.5 * np.exp(-0.5 * ((y - mu) / sigma)**2)

        curve = ParametricFunction(
            lambda t: np.array([vertical_pdf_func(t), t, 0]),
            t_range=[self.grid['E3'][1] + 0.2, max_y + 0.4],
            color=color_line2
        )
        
        # [Issue 36 Fix]: Span curve_label across D5-D6 for better visibility
        curve_label = Text("Weight Distribution", font_size=18, color=color_line2)
        self.place_in_area(curve_label, 'D5', 'D6', scale_factor=0.7)

        self.play(Create(curve), Write(curve_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset previous line color and highlight current line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_line3)
        
        # Shade the "Risk" area (tail of the distribution exceeding max weight)
        tail_points = []
        # Sample points from the curve that are above the max weight line
        for t in np.linspace(max_y, max_y + 0.4, 20):
            tail_points.append(np.array([vertical_pdf_func(t), t, 0]))
        
        # Close the polygon to create a fillable area
        tail_points.append(np.array([self.grid['C2'][0], max_y + 0.4, 0]))
        tail_points.append(np.array([self.grid['C2'][0], max_y, 0]))
        
        shade_area = Polygon(*tail_points, color=color_line3, fill_opacity=0.5, stroke_width=0)
        
        # [Issue 38 Fix]: Center risk_label at A3
        risk_label = Text("Risk", font_size=18, color=color_line3)
        self.place_at_grid(risk_label, 'A3', scale_factor=0.8)

        self.play(FadeIn(shade_area), Write(risk_label))
        self.wait(2)
