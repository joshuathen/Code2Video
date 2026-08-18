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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Inscribed Square Problem", [
            "Can we inscribe a square in any closed loop?",
            "Consider a smooth, continuous, non-self-intersecting curve.",
            "Even for irregular loops, a square often exists."
        ])
        
        # Load SVG
        curve = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/loop.svg", color="#FFFFFF")
        self.place_in_area(curve, 'A3', 'F6', scale_factor=1.5)
        
        # Create quadrilateral vertices tracker
        # Angles: 0, 90, 180, 270 degrees
        angles = [0, PI/2, PI, 3*PI/2]
        dots = VGroup(*[Dot(color="#FF0000") for _ in range(4)])
        
        # Use simple center/radius logic for the loop since its a custom shape
        def get_curve_point(angle):
            # Approximation for the SVG path
            r = 1.0
            return curve.get_center() + r * np.array([np.cos(angle), np.sin(angle), 0])

        def update_dots(d):
            t = self.time
            for i, dot in enumerate(dots):
                angle = angles[i] + t
                dot.move_to(get_curve_point(angle))
        
        quad = VMobject(color="#FF0000")
        quad.add_updater(lambda m: m.set_points_smoothly([d.get_center() for d in dots] + [dots[0].get_center()]))
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(curve))
        self.lecture[0].set_color("#FFFF00")
        self.add(dots, quad)
        dots.add_updater(update_dots)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        # Change quad color to indicate square condition
        quad.set_color("#00FF00")
        for dot in dots:
            dot.set_color("#00FF00")
            
        self.wait(3)
        dots.remove_updater(update_dots)
