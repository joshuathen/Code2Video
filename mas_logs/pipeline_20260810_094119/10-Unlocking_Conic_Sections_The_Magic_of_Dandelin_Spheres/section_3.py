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
        self.setup_layout("Visualizing the Ellipse", [
            "The plane cuts through one cone nappe.",
            "Two spheres touch the cone and plane.",
            "Distance sum to tangency points remains constant."
        ])
        
        # Setup visualization elements
        ellipse = Ellipse(width=3, height=2, color=BLUE)
        f1 = Dot(color=RED)
        f2 = Dot(color=RED)
        
        # Applying requested layout changes
        self.place_in_area(ellipse, 'B3', 'D5', scale_factor=0.9)
        self.place_at_grid(f1, 'B2', scale_factor=0.6)
        self.place_at_grid(f2, 'B4', scale_factor=0.6)
        
        # Load asset
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        self.place_at_grid(sphere_asset, 'C5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # The plane cuts through one cone nappe.
        self.lecture[0].set_color("#FFFFFF")
        self.add(ellipse, sphere_asset)

        # === Animation for Lecture Line 2 ===
        # Two spheres touch the cone and plane.
        self.lecture[1].set_color("#FFFFFF")
        self.add(f1, f2)

        # === Animation for Lecture Line 3 ===
        # Distance sum to tangency points remains constant.
        self.lecture[2].set_color("#FF9900")
        
        p = Dot(color=YELLOW)
        self.place_at_grid(p, 'A3', scale_factor=0.5)
        
        # Use updaters for lines
        line1 = Line(p.get_center(), f1.get_center(), color=WHITE)
        line2 = Line(p.get_center(), f2.get_center(), color=WHITE)
        
        line1.add_updater(lambda m: m.put_start_and_end_on(p.get_center(), f1.get_center()))
        line2.add_updater(lambda m: m.put_start_and_end_on(p.get_center(), f2.get_center()))
        
        self.add(line1, line2, p)
        
        # Animate point moving
        self.play(MoveAlongPath(p, ellipse), run_time=3)
        self.wait(1)
