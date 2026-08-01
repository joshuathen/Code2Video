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

class Section3Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title_text = "The Redundancy Trap: Linear Dependence"
        lecture_lines = [
            "Sometimes, vectors provide redundant movement instructions.",
            "If a vector is already reachable, it is dependent.",
            "Adding a dependent vector doesn't expand the span.",
            "Linear dependence means one vector is a combination.",
            "These redundant vectors add no new dimensions."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        v_color = "#FFD700"
        w_color = "#00FFFF"
        u_color = "#FF4500"
        span_color = "#FFFFFF"
        dep_color = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Sometimes, vectors provide redundant movement instructions.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Define vectors and plane
        # v = (1, 0.5), w = (0.5, 1)
        v_vec = Arrow(ORIGIN, RIGHT + 0.5*UP, buff=0, color=v_color)
        w_vec = Arrow(ORIGIN, 0.5*RIGHT + UP, buff=0, color=w_color)
        v_label = MathTex("\\vec{v}", color=v_color, font_size=24).next_to(v_vec.get_end(), RIGHT, buff=0.1)
        w_label = MathTex("\\vec{w}", color=w_color, font_size=24).next_to(w_vec.get_end(), UP, buff=0.1)
        
        plane = Rectangle(width=4, height=4, fill_color=span_color, fill_opacity=0.1, stroke_width=1, stroke_color=span_color)
        
        # Group vectors and plane to place them together on the grid
        diagram = VGroup(plane, v_vec, w_vec, v_label, w_label)
        # Fix Issue 23: Expanded diagram area to A2-E6
        self.place_in_area(diagram, "A2", "E6")
        
        self.play(Create(plane), GrowArrow(v_vec), GrowArrow(w_vec), Write(v_label), Write(w_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # If a vector is already reachable, it is dependent.
        self.play(self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color(u_color))
        
        # Vector u = v + w = (1.5, 1.5). Origin is same as v_vec
        u_vec = Arrow(v_vec.get_start(), v_vec.get_start() + 1.5*RIGHT + 1.5*UP, buff=0, color=u_color)
        u_label = MathTex("\\vec{u}", color=u_color, font_size=24).next_to(u_vec.get_end(), UR, buff=0.1)
        
        self.play(GrowArrow(u_vec), Write(u_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Adding a dependent vector doesn't expand the span.
        # Animation 3: Use dashed arrows to show that v + w perfectly matches Vector u.
        self.play(self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color(WHITE))
        
        v_ext = DashedLine(v_vec.get_start(), v_vec.get_end(), color=v_color, stroke_width=4).add_tip()
        w_ext = DashedLine(v_vec.get_end(), u_vec.get_end(), color=w_color, stroke_width=4).add_tip()
        
        self.play(Create(v_ext), Create(w_ext))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Linear dependence means one vector is a combination.
        # Animation 4: Flash Vector u and fade it to indicate redundancy.
        self.play(self.lecture[2].animate.set_color(GRAY), self.lecture[3].animate.set_color(WHITE))
        
        # Fade out combination helpers
        self.play(FadeOut(v_ext), FadeOut(w_ext))
        
        # Flash Vector u and fade it to indicate redundancy
        self.play(Indicate(u_vec, color=u_color), run_time=1)
        self.play(u_vec.animate.set_stroke(opacity=0.5), u_label.animate.set_fill(opacity=0.5))
        
        # Flash the plane to show it remains the same
        self.play(plane.animate.set_fill(opacity=0.4), run_time=0.4)
        self.play(plane.animate.set_fill(opacity=0.1), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # These redundant vectors add no new dimensions.
        self.play(self.lecture[3].animate.set_color(GRAY), self.lecture[4].animate.set_color(dep_color))
        
        dep_set = MathTex("\\{\\vec{v}, \\vec{w}, \\vec{u}\\}", color=WHITE, font_size=28)
        dep_status = Text("Linearly Dependent", color=dep_color, font_size=24)
        dep_box = VGroup(dep_set, dep_status).arrange(DOWN, buff=0.2)
        
        # Fix Issue 24: Positioned dep_box in area F2-F6
        self.place_in_area(dep_box, "F2", "F6")
        
        self.play(Indicate(u_vec, color=dep_color), Write(dep_box))
        self.wait(2)
