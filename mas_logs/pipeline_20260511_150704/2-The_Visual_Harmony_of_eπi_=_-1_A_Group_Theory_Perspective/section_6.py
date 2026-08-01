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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Summary: The Bridge Between Algebra and Geometry", 
            [
                "The complex plane unites growth and rotation.", 
                "Geometry and algebra merge into a single story.", 
                "Euler’s identity remains mathematics' most beautiful balance."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Color: Yellow
        self.lecture[0].set_color(YELLOW)
        
        # Create Complex Plane elements
        plane = ComplexPlane(
            x_range=[-2, 2, 1], 
            y_range=[-2, 2, 1], 
            background_line_style={"stroke_opacity": 0.2}
        )
        unit_circle = Circle(radius=1, color=YELLOW, stroke_width=2)
        
        # Key formula for initial view
        formula = Text("e^πi = -1", color=YELLOW, font_size=42)
        
        # Position Complex Plane in the middle area (Resolved Issue 53)
        plane_group = VGroup(plane, unit_circle)
        self.place_in_area(plane_group, "B1", "E6", scale_factor=0.8)
        
        # Position formula at bottom (Resolved Issue 54 & 55)
        self.place_in_area(formula, "F2", "F5", scale_factor=0.8)
        
        # Whole scene group for the zoom out
        entire_visual = VGroup(plane_group, formula)
        entire_visual.save_state()
        entire_visual.scale(0.1).set_opacity(0)
        
        self.play(
            entire_visual.animate.scale(10).set_opacity(1),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color: Teal
        self.lecture[1].set_color(TEAL)
        
        # Asset: Bridge (Resolved Issue 38)
        bridge_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/bridge.svg")
        bridge_icon.set_color(TEAL).set_opacity(0.3)
        self.place_in_area(bridge_icon, "D2", "E5", scale_factor=1.2)
        
        # Labels
        label_calc = Text("Calculus", font_size=18, color=TEAL)
        label_geom = Text("Geometry", font_size=18, color=TEAL)
        label_group = Text("Group Theory", font_size=18, color=TEAL)
        
        # Position labels around the central formula
        self.place_at_grid(label_calc, "E2", scale_factor=1.0)
        self.place_at_grid(label_geom, "E4", scale_factor=1.0)
        self.place_at_grid(label_group, "E6", scale_factor=1.0)
        
        labels = VGroup(label_calc, label_geom, label_group)
        
        self.play(FadeIn(bridge_icon), FadeIn(labels))
        self.play(
            Indicate(labels, color=TEAL, scale_factor=1.2),
            Flash(formula, color=TEAL, line_length=0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color: Gold
        self.lecture[2].set_color("#FFD700")
        
        # Transformation of equation
        new_formula = Text("e^πi + 1 = 0", color="#FFD700", font_size=42)
        # Use same layout logic for new formula (Resolved Issue 54 & 55)
        self.place_in_area(new_formula, "F2", "F5", scale_factor=0.8)
        
        self.play(
            Transform(formula, new_formula),
            formula.animate.set_color("#FFD700"),
            run_time=1.5
        )
        
        # Glow effect (Indicate + color update)
        self.play(
            Indicate(formula, color="#FFD700", scale_factor=1.1),
            formula.animate.set_stroke(width=1, opacity=0.8),
            run_time=2
        )
        
        self.wait(3)
