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
            "Point P sits on our ellipse.",
            "Distances to two foci add up.",
            "Sum remains constant for every point.",
            "Tangent properties prove this sum invariant.",
            "Dandelin Spheres confirm the ellipse property."
        ]
        self.setup_layout("Proving the Ellipse Property", lecture_lines)
        
        # Adjust layout based on feedback
        self.place_in_area(self.title, 'A1', 'A6', scale_factor=1.0)
        self.place_in_area(self.lecture, 'B1', 'D3', scale_factor=0.85)
        
        # Ellipse and Foci setup
        ellipse = Ellipse(width=3, height=2, color=WHITE)
        f1 = Dot(color=BLUE)
        f2 = Dot(color=BLUE)
        f1.move_to(ellipse.get_center() + LEFT * 0.8)
        f2.move_to(ellipse.get_center() + RIGHT * 0.8)
        
        p = Dot(color=YELLOW)
        p.move_to(ellipse.point_from_proportion(0.2))
        
        # Asset Loading
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=YELLOW)
        sphere_asset.scale(0.3)
        sphere_asset.move_to(p.get_center())
        
        path1 = Line(p.get_center(), f1.get_center(), color=BLUE_A)
        path2 = Line(p.get_center(), f2.get_center(), color=BLUE_B)
        
        ellipse_group = VGroup(ellipse, f1, f2, p, path1, path2, sphere_asset)
        self.place_in_area(ellipse_group, 'B4', 'E6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(ellipse), Create(p), FadeIn(sphere_asset))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(Create(f1), Create(f2), Create(path1), Create(path2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # Updater for sphere and lines
        def update_group(m):
            p_pos = p.get_center()
            path1.become(Line(p_pos, f1.get_center(), color=BLUE_A))
            path2.become(Line(p_pos, f2.get_center(), color=BLUE_B))
            sphere_asset.move_to(p_pos)
            
        ellipse_group.add_updater(update_group)
        self.play(MoveAlongPath(p, ellipse), run_time=3)
        ellipse_group.remove_updater(update_group)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(PURPLE))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(ORANGE))
        self.play(ellipse.animate.set_stroke(color=ORANGE, width=6))
