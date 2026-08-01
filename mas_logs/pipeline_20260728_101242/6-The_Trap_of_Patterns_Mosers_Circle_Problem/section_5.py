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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "The Geometric Secret: Why 31?"
        lecture_lines = [
            "The real formula depends on chords and intersections.",
            "Every internal crossing comes from four boundary points.",
            "We use combinations to count these geometric features.",
            "For six points, the formula gives thirty-one regions.",
            "Math provides a deeper explanation than simple observation."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors from storyboard
        c1 = "#FF0000"
        c2 = "#FFFF00"
        c3 = "#00FFFF"
        c4 = "#00FF00"
        c5 = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight 4 specific points [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/points.svg] (Dot, #FF0000) 
        # on the circle [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/boundary.svg] (Circle, #FFFFFF).
        self.play(self.lecture[0].animate.set_color(c1))
        
        # Math helper for point calculation (not added to scene)
        circle_math = Circle(radius=1.4)
        
        # Boundary SVG Asset
        circle_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/boundary.svg").set_color(WHITE)
        self.place_in_area(circle_svg, "B2", "E5")
        
        # Align math circle to SVG to get point positions
        circle_math.move_to(circle_svg.get_center()).scale_to_fit_width(circle_svg.width)
        
        # 4 points for the demonstration
        p1 = circle_math.point_at_angle(60 * DEGREES)
        p2 = circle_math.point_at_angle(150 * DEGREES)
        p3 = circle_math.point_at_angle(240 * DEGREES)
        p4 = circle_math.point_at_angle(330 * DEGREES)
        
        # Points SVG Asset
        point_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/points.svg"
        dots = VGroup(
            SVGMobject(point_asset).set_color(c1).scale(0.15).move_to(p1),
            SVGMobject(point_asset).set_color(c1).scale(0.15).move_to(p2),
            SVGMobject(point_asset).set_color(c1).scale(0.15).move_to(p3),
            SVGMobject(point_asset).set_color(c1).scale(0.15).move_to(p4)
        )
        
        self.play(FadeIn(circle_svg))
        self.play(FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the two chords connecting them crossing to form 1 intersection (Dot, #FFFF00).
        self.play(self.lecture[1].animate.set_color(c2))
        
        chord1 = Line(p1, p3, color=c2)
        chord2 = Line(p2, p4, color=c2)
        # Center of the symmetric points is the intersection
        intersection = Dot(circle_svg.get_center(), color=c2, radius=0.1)
        
        self.play(Create(chord1), Create(chord2))
        self.play(FadeIn(intersection))
        self.play(Flash(intersection, color=c2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the term binom(n, 4) next to the intersection (Text, #00FFFF).
        self.play(self.lecture[2].animate.set_color(c3))
        
        binom4 = MathTex(r"\binom{n}{4}", color=c3)
        # Fix overlapping and scale based on VideoCritic issues #34 and #35
        self.place_at_grid(binom4, "C6", scale_factor=0.8)
        
        self.play(Write(binom4))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display the full formula R(n) = binom(n, 4) + binom(n, 2) + 1 (Text, #00FF00).
        self.play(self.lecture[3].animate.set_color(c4))
        
        formula = MathTex(r"R(n) = \binom{n}{4} + \binom{n}{2} + 1", color=c4)
        self.place_in_area(formula, "A1", "A6", scale_factor=0.9)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Calculate the formula for n=6 to show it equals 31 (Text, #FFFFFF).
        self.play(self.lecture[4].animate.set_color(c5))
        
        calc_res = MathTex(
            r"R(6) = \binom{6}{4} + \binom{6}{2} + 1 = 15 + 15 + 1 = 31",
            color=c5
        )
        self.place_in_area(calc_res, "F1", "F6", scale_factor=0.8)
        
        self.play(Write(calc_res))
        self.play(Indicate(calc_res))
        self.wait(2)
