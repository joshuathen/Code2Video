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
        # Setup the layout with the specific title and lecture lines for Section 1
        title_text = "The Paradox of Turbulence"
        lecture_lines = [
            "Turbulence looks chaotic but hides mathematical order.",
            "Smooth laminar flow breaks into complex swirling eddies.",
            "We seek the constants governing this magnificent mess."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors
        COLOR_LAMINAR = "#0000FF"    # Blue
        COLOR_WING = "#808080"       # Gray
        COLOR_EDDY = "#FFFFFF"       # White
        COLOR_SMALL_EDDY = "#FFA500" # Orange
        COLOR_LABEL = "#FFFF00"      # Yellow

        # === Animation for Lecture Line 1 ===
        # Highlight line 1 and draw smooth blue streamlines.
        self.play(self.lecture[0].animate.set_color(COLOR_LAMINAR))
        
        # Create horizontal streamlines in the animation area
        streamlines = VGroup(*[
            Line(LEFT * 2.5, RIGHT * 2.5, color=COLOR_LAMINAR, stroke_width=2)
            for _ in range(7)
        ]).arrange(DOWN, buff=0.4)
        # Apply Fix for Issue 27: Reduce vertical area to focus on wake interaction
        self.place_in_area(streamlines, "B2", "E5", scale_factor=1.0)
        
        self.play(Create(streamlines), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight to line 2. Introduce wing profile and transform flow into eddies.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_EDDY)
        )
        
        # Wing profile (Gray)
        wing = Ellipse(width=1.8, height=0.6, color=COLOR_WING, fill_opacity=1).set_fill(COLOR_WING)
        # Apply Fix for Issue 28: Scale down wing for better transition room
        self.place_at_grid(wing, "C2", scale_factor=0.7)
        
        # Define curved streamlines that deviate around the wing
        curved_streamlines = VGroup()
        for i in range(len(streamlines)):
            y_off = (i - 3) * 0.4
            
            p1 = self.grid["C1"] + UP * y_off + LEFT * 0.5
            p4 = self.grid["C6"] + UP * y_off + RIGHT * 0.5
            
            if abs(y_off) < 0.8:
                curve_dir = 0.6 if y_off >= 0 else -0.6
                p2 = self.grid["C2"] + UP * (y_off + curve_dir)
                p3 = self.grid["C4"] + UP * y_off
                path = VMobject(color=COLOR_LAMINAR).set_points_as_corners([p1, p2, p3, p4]).make_smooth()
            else:
                path = Line(p1, p4, color=COLOR_LAMINAR)
            curved_streamlines.add(path)

        self.play(
            FadeIn(wing),
            ReplacementTransform(streamlines, curved_streamlines),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Transform curved flow into white swirling eddies
        eddies = VGroup()
        eddy_spots = ["B4", "C4", "D4", "C5", "D5"]
        for spot in eddy_spots:
            circ = Circle(radius=0.25, color=COLOR_EDDY)
            self.place_at_grid(circ, spot)
            swirl = Arc(radius=0.25, start_angle=0, angle=250*DEGREES, color=COLOR_EDDY).add_tip(tip_length=0.1)
            swirl.move_to(circ.get_center())
            eddies.add(VGroup(circ, swirl))
            
        self.play(
            ReplacementTransform(curved_streamlines, eddies),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight to line 3. Add complexity and the turbulence label.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_SMALL_EDDY)
        )
        
        # Small orange swirls for complexity
        small_eddies = VGroup()
        # Adjusted spots to avoid overlap with new label position at A5
        small_eddy_spots = ["A4", "A6", "B6", "D6", "E6", "F6", "F5", "F4"]
        for spot in small_eddy_spots:
            se = Circle(radius=0.12, color=COLOR_SMALL_EDDY)
            self.place_at_grid(se, spot)
            small_eddies.add(se)
            
        self.play(LaggedStartMap(FadeIn, small_eddies, lag_ratio=0.1))
        
        # Highlight one eddy and label it 'Turbulence' (Yellow)
        target_eddy = eddies[3] # Eddy at C5
        label = Text("Turbulence", font_size=24, color=COLOR_LABEL)
        # Apply Fix for Issue 26: Move label to A5 and scale down to reduce clutter
        self.place_at_grid(label, "A5", scale_factor=0.8) 
        
        self.play(
            target_eddy.animate.set_color(COLOR_LABEL).scale(1.2),
            Write(label),
            self.lecture[2].animate.set_color(COLOR_LABEL)
        )
        self.wait(3)
