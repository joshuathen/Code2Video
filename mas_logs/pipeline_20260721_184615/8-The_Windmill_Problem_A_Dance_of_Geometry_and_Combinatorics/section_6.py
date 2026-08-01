from manim import *
import numpy as np

# === Base TeachingScene Class ===
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

# === Section 6 Scene ===
class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize Layout
        title_text = "Conclusion and Application"
        lecture_lines = [
            "- Local rules create a beautiful, predictable global cycle.",
            "- This visual logic solved the hardest IMO 2011 problem.",
            "- Simple rotations reveal deep truths in geometry and combinatorics."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a fast-motion animation of a 10-point windmill [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg]
        # completing a full cycle. self.wait(2.0).
        
        # Highlight first lecture line
        self.lecture[0].set_color("#00FFFF") # Cyan
        
        # Asset: windmill.svg icon
        windmill_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        self.place_at_grid(windmill_icon, "A1", scale_factor=0.3)
        self.add(windmill_icon)
        
        # Define 10 points in the visual area (B2 to F6)
        point_coords = [
            self.grid["B3"] + [0.2, 0.3, 0],
            self.grid["B5"] + [-0.1, -0.2, 0],
            self.grid["C2"] + [0.4, 0.1, 0],
            self.grid["C5"] + [-0.3, 0.5, 0],
            self.grid["D3"] + [0.5, -0.4, 0],
            self.grid["D6"] + [-0.2, 0.2, 0],
            self.grid["E2"] + [0.3, 0.6, 0],
            self.grid["E4"] + [0.1, -0.3, 0],
            self.grid["F3"] + [-0.4, 0.2, 0],
            self.grid["F5"] + [-0.1, -0.5, 0],
        ]
        points = VGroup(*[Dot(coord, color="#00FFFF", radius=0.08) for coord in point_coords])
        self.add(points)
        
        # Initial pivot point
        pivot_idx = 2 # Start at C2
        points[pivot_idx].set_color("#FF00FF") # Magenta for pivot
        
        # Rotating line
        line = Line(points[pivot_idx].get_center() - 2.5*RIGHT, points[pivot_idx].get_center() + 2.5*RIGHT, color="#FFFFFF")
        line.set_stroke(width=2)
        self.add(line)
        
        # Fast-motion pivot sequence simulation
        pivots_seq = [5, 7, 1, 9, 0] # Sequence of indices for "hitting" points
        for next_p in pivots_seq:
            # Rotate around current pivot
            self.play(Rotate(line, angle=PI/2, about_point=points[pivot_idx].get_center()), run_time=0.3, rate_func=linear)
            # Switch pivot "instantly"
            self.play(
                points[pivot_idx].animate.set_color("#00FFFF"),
                points[next_p].animate.set_color("#FF00FF"),
                line.animate.move_to(points[next_p].get_center()),
                run_time=0.1
            )
            pivot_idx = next_p
            
        # Final rotation to finish the "cycle"
        self.play(Rotate(line, angle=PI, about_point=points[pivot_idx].get_center()), run_time=0.5, rate_func=linear)
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Display the text 'IMO 2011 Problem 2' in gold (#FFD700) alongside a medal icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/medal.svg].
        
        # Highlight second lecture line and revert first
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700"), # Gold
            run_time=0.5
        )
        
        # Fix Issue 35: Position label in area A1-A6
        imo_text = Text("IMO 2011 Problem 2", color="#FFD700", font_size=32)
        self.place_in_area(imo_text, "A1", "A6", scale_factor=0.7)
        
        # Asset: medal.svg icon
        medal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/medal.svg")
        medal_icon.scale(0.3)
        medal_icon.next_to(imo_text, LEFT, buff=0.2)
        
        self.play(
            Write(imo_text),
            FadeIn(medal_icon)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Fade out the points and line into a final geometric pattern. self.wait(2.0).
        
        # Highlight third lecture line and revert second
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00"), # Green
            run_time=0.5
        )
        
        # Create a final geometric pattern (Decagon with diagonals)
        pattern_inner = RegularPolygon(n=10, color="#00FF00")
        # Fix Issue 36: Position pattern in area B1-F6
        self.place_in_area(pattern_inner, "B1", "F6", scale_factor=0.8)
        
        # Adding diagonals
        vertices = pattern_inner.get_vertices()
        diagonals = VGroup()
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                diagonals.add(Line(vertices[i], vertices[j], stroke_width=0.5, color="#00FF00", stroke_opacity=0.4))
        
        final_visual = VGroup(pattern_inner, diagonals)
        
        # Fade out elements and show final pattern
        self.play(
            FadeOut(points),
            FadeOut(line),
            FadeOut(imo_text),
            FadeOut(medal_icon),
            FadeOut(windmill_icon),
            FadeIn(final_visual),
            run_time=1.5
        )
        
        # Final indication - [L004] use Indicate
        self.play(Indicate(final_visual, color="#00FF00", scale_factor=1.1))
        
        self.wait(2.0)
