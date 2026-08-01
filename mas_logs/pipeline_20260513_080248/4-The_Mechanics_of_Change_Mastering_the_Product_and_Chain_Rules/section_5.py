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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial setup
        lecture_lines = [
            "Product rule adds the growth of two combined parts.",
            "Chain rule multiplies the links of nested functions.",
            "Master these mechanics to solve any complex derivative."
        ]
        self.setup_layout("Visual Summary and Cheat Sheet", lecture_lines)

        # === Pre-creation of objects ===
        
        # Product Rule Elements
        plus_h = Rectangle(height=0.15, width=0.8, fill_opacity=1, color="#FFFF00", stroke_width=0)
        plus_v = Rectangle(height=0.8, width=0.15, fill_opacity=1, color="#FFFF00", stroke_width=0)
        plus_icon = VGroup(plus_h, plus_v)
        
        # Unicode prime used for derivatives: '
        prod_formula = Text("(uv)' = uv' + vu'", font_size=36, color=WHITE)
        prod_title = Text("Product Rule", font_size=24, color="#FFFF00")
        
        # Chain Rule Elements
        # Simple Link Icon (two interlocking rings)
        link1 = Annulus(inner_radius=0.25, outer_radius=0.35, color="#00FFFF")
        link2 = Annulus(inner_radius=0.25, outer_radius=0.35, color="#00FFFF").shift(RIGHT * 0.4)
        links_icon = VGroup(link1, link2)
        
        # Composition symbol: \u2218
        chain_formula = Text("(f \u2218 g)' = f'(g) \u00B7 g'", font_size=36, color=WHITE)
        chain_title = Text("Chain Rule", font_size=24, color="#00FFFF")

        # Positioning
        self.place_in_area(prod_title, "A1", "A3", scale_factor=1.0)
        self.place_in_area(plus_icon, "B1", "C3", scale_factor=0.7)
        self.place_in_area(prod_formula, "D1", "E3", scale_factor=0.6)
        
        self.place_in_area(chain_title, "A4", "A6", scale_factor=1.0)
        self.place_in_area(links_icon, "B4", "C6", scale_factor=0.7)
        self.place_in_area(chain_formula, "D4", "E6", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        # Highlight first line and show Product Rule
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=0.5)
        self.play(
            FadeIn(prod_title),
            GrowFromCenter(plus_icon),
            Write(prod_formula),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight second line and show Chain Rule
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF"),
            run_time=0.5
        )
        self.play(
            FadeIn(chain_title),
            GrowFromCenter(links_icon),
            Write(chain_formula),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Master these mechanics... Final Reveal
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE),
            run_time=0.5
        )
        
        # Fade out everything for the final message
        all_graphics = VGroup(
            prod_title, plus_icon, prod_formula,
            chain_title, links_icon, chain_formula,
            self.lecture, self.title
        )
        
        final_msg = Text("The Mechanics of Change", font_size=48, color=WHITE)
        # Using a coordinate plane as a moving background
        plane = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).fade(0.7)

        self.play(
            FadeOut(all_graphics),
            FadeIn(plane),
            run_time=1.5
        )
        
        # Center the final message in the visual area
        # Issue 35 fix: Use area B2 to E5 and scale 0.8
        self.place_in_area(final_msg, "B2", "E5", scale_factor=0.8)
        
        self.play(Write(final_msg), run_time=2)
        
        # Slow movement of the plane for the fade out effect
        self.play(
            plane.animate.shift(RIGHT * 1.5 + UP * 0.5),
            final_msg.animate.scale(1.05),
            run_time=4,
            rate_func=linear
        )
        
        self.wait(1)
