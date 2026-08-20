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
            "Moving to denser media, light bends inward.",
            "Moving to lighter media, light bends outward.",
            "Always measure angles from the normal line."
        ]
        self.setup_layout("Visualizing Direction: Toward or Away from Normal", lecture_lines)
        
        # Setup visuals
        # Need to place diagrams carefully using area grid
        glass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        water_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg")
        
        # Ray Diagrams
        ray1 = VGroup(
            DashedLine(UP*1.5, DOWN*1.5, color=GREY),
            Line(LEFT*1.5, ORIGIN, color=YELLOW),
            Line(ORIGIN, DOWN*1, color=YELLOW).rotate(-0.3, about_point=ORIGIN)
        )
        
        ray2 = VGroup(
            DashedLine(UP*1.5, DOWN*1.5, color=GREY),
            Line(UP*1, ORIGIN, color=YELLOW).rotate(0.3, about_point=ORIGIN),
            Line(ORIGIN, DOWN*1.5, color=YELLOW).rotate(0.6, about_point=ORIGIN)
        )
        
        diagrams = VGroup(
            VGroup(glass_icon, ray1).arrange(DOWN),
            VGroup(water_icon, ray2).arrange(DOWN)
        ).arrange(RIGHT, buff=1.0)
        
        self.place_in_area(diagrams, 'A1', 'F3', scale_factor=0.9)
        
        # Normal label
        normal_label = Text("Normal", font_size=24, color=GREY)
        self.place_at_grid(normal_label, 'C4', scale_factor=0.7)
        
        # Instructional text
        instructional_text = Text("Measure from normal", font_size=20, color=WHITE)
        self.place_in_area(instructional_text, 'D1', 'E3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(diagrams[0]))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(FadeIn(diagrams[1]))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        angle_arc = Arc(radius=0.5, start_angle=PI/2, angle=0.5, color="#FF4500")
        angle_arc.move_to(ray1[0].get_center() + RIGHT*0.2 + DOWN*0.2)
        self.play(Create(angle_arc), FadeIn(normal_label), FadeIn(instructional_text))
        self.wait(2)
