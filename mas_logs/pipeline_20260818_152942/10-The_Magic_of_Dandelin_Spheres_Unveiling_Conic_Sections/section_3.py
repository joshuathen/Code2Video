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
        self.setup_layout("The Geometry of the Ellipse", [
            "An ellipse forms by slicing one cone.",
            "Points on the ellipse follow specific rules.",
            "The sum of distances is always constant.",
            "Foci are where the spheres touch.",
            "This creates the classic elliptical shape."
        ])
        
        # Ellipse representation
        ellipse = Ellipse(width=3, height=2, color=ORANGE, stroke_width=4)
        self.place_at_grid(ellipse, 'C4', scale_factor=0.8)

        # Foci
        f1 = Dot(color=YELLOW)
        f2 = Dot(color=YELLOW)
        self.place_at_grid(f1, 'B3', scale_factor=0.5)
        self.place_at_grid(f2, 'D5', scale_factor=0.5)

        # Dandelin spheres
        sphere1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        self.place_at_grid(sphere1, 'B4', scale_factor=0.3)
        self.place_at_grid(sphere2, 'D4', scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.play(Create(ellipse), self.lecture[0].animate.set_color(ORANGE))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(sphere1), FadeIn(sphere2), self.lecture[1].animate.set_color(YELLOW))
        
        # Point P on ellipse
        p = Dot(color=BLUE)
        p.move_to(ellipse.point_from_proportion(0.2))
        self.add(p)

        # Lines to foci (using persistent objects and ValueTracker to update rather than always_redraw)
        l1 = Line(p.get_center(), f1.get_center(), color=BLUE, stroke_width=2)
        l2 = Line(p.get_center(), f2.get_center(), color=BLUE, stroke_width=2)
        self.add(l1, l2)

        def update_lines(mob):
            l1.put_start_and_end_on(p.get_center(), f1.get_center())
            l2.put_start_and_end_on(p.get_center(), f2.get_center())
        
        p.add_updater(update_lines)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.play(MoveAlongPath(p, ellipse), run_time=3, rate_func=linear)
        p.remove_updater(update_lines)

        # === Animation for Lecture Line 4 ===
        foci_label = Text("Foci", font_size=18, color=YELLOW)
        self.place_at_grid(foci_label, 'C5', scale_factor=0.6)
        self.play(Write(foci_label), self.lecture[3].animate.set_color(YELLOW))

        # === Animation for Lecture Line 5 ===
        self.play(Flash(ellipse, color=ORANGE, num_lines=10), self.lecture[4].animate.set_color(ORANGE))
        self.wait(1)
