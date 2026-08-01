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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Imagine two vectors spanning a full flat plane.',
            'Add a third vector already landing within that span.',
            'This third vector is just a mix of others.',
            'We call this redundancy linear dependence.',
            'The reachable span remains exactly the same size.'
        ]
        self.setup_layout("Linear Dependence: The Redundant Helper", lecture_lines)

        # Vector colors
        V1_COLOR = "#00FF00" # Green
        V2_COLOR = "#0000FF" # Blue
        V3_COLOR = "#FFFF00" # Yellow
        DEPENDENT_COLOR = "#FF0000" # Red
        SPAN_COLOR = "#222222" # Dark Grey

        # Origin for vectors
        origin = self.grid["D3"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Define v1 and v2
        v1_end = self.grid["C2"]
        v2_end = self.grid["E2"]
        v1 = Arrow(origin, v1_end, buff=0, color=V1_COLOR)
        v2 = Arrow(origin, v2_end, buff=0, color=V2_COLOR)
        
        v1_label = Text("v1", font_size=18, color=V1_COLOR)
        self.place_at_grid(v1_label, "C2")
        
        v2_label = Text("v2", font_size=18, color=V2_COLOR)
        self.place_at_grid(v2_label, "E2")

        # Shaded span (covering the background of the grid area)
        span_poly = Polygon(
            self.grid["A1"], self.grid["A6"], self.grid["F6"], self.grid["F1"],
            fill_color=SPAN_COLOR, fill_opacity=0.5, stroke_width=0
        )
        
        self.play(FadeIn(span_poly))
        self.play(GrowArrow(v1), GrowArrow(v2))
        self.play(Write(v1_label), Write(v2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # v3 points to the sum of v1 and v2 (vector addition logic)
        # Vector D3->C2 is [-1, 1]. D3->E2 is [-1, -1]. Sum relative to D3 is [-2, 0].
        # D3 is row D, col 3. D1 is row D, col 1.
        v3_end = self.grid["D1"]
        v3 = Arrow(origin, v3_end, buff=0, color=V3_COLOR)
        v3_label = Text("v3", font_size=18, color=V3_COLOR)
        self.place_at_grid(v3_label, "D2", scale_factor=0.8)

        self.play(GrowArrow(v3), Write(v3_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Ghost components to show v3 = v1 + v2
        ghost_v1 = Arrow(origin, v1_end, buff=0, color=V1_COLOR).set_stroke(opacity=0.5)
        ghost_v2 = Arrow(v1_end, v3_end, buff=0, color=V2_COLOR).set_stroke(opacity=0.5)

        self.play(FadeIn(ghost_v1), FadeIn(ghost_v2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        # Highlight v3 as redundant
        dep_text = Text("Linearly Dependent", font_size=20, color=DEPENDENT_COLOR)
        self.place_in_area(dep_text, "E4", "F6")

        self.play(v3.animate.set_color(DEPENDENT_COLOR), v3_label.animate.set_color(DEPENDENT_COLOR))
        self.play(Write(dep_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        
        # Remove v3 and ghosts, show span is the same
        self.play(FadeOut(v3), FadeOut(v3_label), FadeOut(ghost_v1), FadeOut(ghost_v2), FadeOut(dep_text))
        
        # Flash the span to show it stayed the same
        self.play(span_poly.animate.set_fill(opacity=0.8), run_time=0.5)
        self.play(span_poly.animate.set_fill(opacity=0.5), run_time=0.5)
        
        self.wait(2)
