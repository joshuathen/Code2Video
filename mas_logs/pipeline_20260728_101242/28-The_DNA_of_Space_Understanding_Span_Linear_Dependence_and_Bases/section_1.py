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
        self.setup_layout(
            "The Starting Kit: Prerequisites and Linear Combinations",
            [
                "Vectors represent instructions for movement in space.",
                "We can scale vectors to change their reach.",
                "Adding these scaled vectors creates a linear combination."
            ]
        )
        
        # Origin at E2 to allow for expansion in North (Up) and East (Right) directions.
        origin_pos = 'E2'
        origin = self.grid[origin_pos]
        
        # === Animation for Lecture Line 1 ===
        # Vectors represent instructions for movement in space.
        self.lecture[0].set_color("#FFD700")
        
        # Compass Asset to establish directional context
        # Resolved Issue 18: Move compass origin to E2
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, origin_pos, scale_factor=0.6)
        compass.set_opacity(0.4)
        
        v_color = "#FFD700"
        w_color = "#00FFFF"
        
        # Vector v: East (E2 to E3)
        v_arrow = Arrow(origin, self.grid['E3'], buff=0, color=v_color)
        # Resolved Issue 19: Move v_label to F3
        v_label = Text("v: East", font_size=16, color=v_color)
        self.place_at_grid(v_label, 'F3', scale_factor=1.0)
        
        # Vector w: North (E2 to D2)
        w_arrow = Arrow(origin, self.grid['D2'], buff=0, color=w_color)
        # Resolved Issue 20: Move w_label to D1
        w_label = Text("w: North", font_size=16, color=w_color)
        self.place_at_grid(w_label, 'D1', scale_factor=1.0)
        
        self.play(FadeIn(compass))
        self.play(Create(v_arrow), Write(v_label))
        self.play(Create(w_arrow), Write(w_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We can scale vectors to change their reach.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFACD")
        
        v_scaled_color = "#FFFACD"
        w_scaled_color = "#E0FFFF"
        
        # Scale v by 2 (E2 to E4)
        v2_arrow = Arrow(origin, self.grid['E4'], buff=0, color=v_scaled_color)
        # Resolved Issue 19: Move v2_label to F4
        v2_label = Text("2v", font_size=16, color=v_scaled_color)
        self.place_at_grid(v2_label, 'F4', scale_factor=1.0)
        
        # Scale w by 3 (E2 to B2)
        w3_arrow = Arrow(origin, self.grid['B2'], buff=0, color=w_scaled_color)
        # Resolved Issue 20: Move w3_label to B1
        w3_label = Text("3w", font_size=16, color=w_scaled_color)
        self.place_at_grid(w3_label, 'B1', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(v_arrow.copy(), v2_arrow),
            Write(v2_label),
            v_arrow.animate.set_opacity(0.3)
        )
        self.play(
            ReplacementTransform(w_arrow.copy(), w3_arrow),
            Write(w3_label),
            w_arrow.animate.set_opacity(0.3)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Adding these scaled vectors creates a linear combination.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#ADFF2F")
        
        # Tip-to-tail: move w3_arrow so tail is at v2_arrow tip (E4)
        # Resulting tip is B4 (since B is 3 units north of E)
        w3_moved = Arrow(self.grid['E4'], self.grid['B4'], buff=0, color=w_scaled_color)
        
        self.play(
            ReplacementTransform(w3_arrow, w3_moved),
            FadeOut(w3_label)
        )
        
        # Draw resultant (E2 to B4)
        res_color = "#ADFF2F"
        res_arrow = Arrow(origin, self.grid['B4'], buff=0, color=res_color)
        res_label = Text("2v + 3w", font_size=18, color=res_color)
        self.place_at_grid(res_label, 'B5', scale_factor=1.0)
        
        self.play(Create(res_arrow), Write(res_label))
        
        # Highlight final point at B4
        final_dot = Dot(self.grid['B4'], color=WHITE)
        flash_circle = Circle(radius=0.1, color=WHITE).move_to(self.grid['B4'])
        
        self.play(FadeIn(final_dot))
        self.play(
            flash_circle.animate.scale(5).set_opacity(0),
            run_time=1,
            rate_func=rate_functions.ease_out_quad
        )
        self.wait(2)
