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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        self.setup_layout("The Hook: The Lifeguard's Dilemma", [
            "A lifeguard must reach a swimmer as fast as possible.",
            "Running on sand is faster than swimming in water.",
            "The quickest path isn't straight, but Fermat's way."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight line 1
        self.lecture[0].set_color(WHITE)
        
        # Draw background: sand (top half) and water (bottom half)
        # The grid rows A-C are sand, D-F are water. Midpoint is between C and D.
        sand_bg = Rectangle(width=5, height=3, fill_color="#C2B280", fill_opacity=0.4, stroke_width=0)
        self.place_in_area(sand_bg, 'A1', 'C6')
        
        water_bg = Rectangle(width=5, height=3, fill_color="#0077BE", fill_opacity=0.4, stroke_width=0)
        self.place_in_area(water_bg, 'D1', 'F6')
        
        # Place Point L (Lifeguard) at top-left and Point S (Swimmer) at bottom-right
        dot_l = Dot(color="#FFFFFF")
        self.place_at_grid(dot_l, 'A1')
        label_l = Text("L", font_size=20, color="#FFFFFF")
        label_l.next_to(dot_l, UP, buff=0.1)
        
        dot_s = Dot(color="#FFFFFF")
        self.place_at_grid(dot_s, 'F6')
        label_s = Text("S", font_size=20, color="#FFFFFF")
        label_s.next_to(dot_s, DOWN, buff=0.1)
        
        self.play(FadeIn(sand_bg), FadeIn(water_bg))
        self.play(Create(dot_l), Write(label_l), Create(dot_s), Write(label_s))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight line 2 with yellow-gold to represent sand speed focus
        self.lecture[1].set_color("#FFD700")
        
        # Display velocity vector arrows
        v_sand_arrow = Arrow(start=LEFT*0.8, end=RIGHT*0.8, color="#FFD700", buff=0)
        self.place_at_grid(v_sand_arrow, 'B3')
        v_sand_label = Text("v_sand", font_size=18, color="#FFD700")
        v_sand_label.next_to(v_sand_arrow, UP, buff=0.1)
        
        v_water_arrow = Arrow(start=LEFT*0.4, end=RIGHT*0.4, color="#1E90FF", buff=0)
        self.place_at_grid(v_water_arrow, 'E3')
        v_water_label = Text("v_water", font_size=18, color="#1E90FF")
        v_water_label.next_to(v_water_arrow, UP, buff=0.1)
        
        self.play(GrowArrow(v_sand_arrow), Write(v_sand_label))
        self.play(GrowArrow(v_water_arrow), Write(v_water_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight line 3 with green to represent the optimal path
        self.lecture[2].set_color("#00FF00")
        
        # Straight path (dashed)
        straight_path = DashedLine(dot_l.get_center(), dot_s.get_center(), color="#888888")
        
        # Bent path (Optimal: spends more time on sand)
        # Interface point is calculated at the boundary between Row C and D (y = -0.3)
        interface_x = self.grid['C5'][0]
        interface_y = (self.grid['C5'][1] + self.grid['D5'][1]) / 2
        interface_point = np.array([interface_x, interface_y, 0])
        
        bent_path = VMobject(color="#00FF00")
        bent_path.set_points_as_corners([dot_l.get_center(), interface_point, dot_s.get_center()])
        
        self.play(Create(straight_path))
        self.play(Create(bent_path), run_time=2)
        self.wait(2)
