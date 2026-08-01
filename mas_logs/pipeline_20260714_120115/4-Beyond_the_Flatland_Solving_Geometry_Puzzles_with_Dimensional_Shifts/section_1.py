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
        title = "The Boundary Constraint: Why Dimensions Matter"
        lecture_lines = [
            "Some geometry problems seem impossible within their own dimension.",
            "Imagine an ant trapped inside a simple 2D circle.",
            "In 3D, we can easily step over this boundary."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors for coordination
        color_1 = WHITE
        color_2 = RED
        color_3 = BLUE_B # Light blue

        # Assets
        ant_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/ant.svg"
        camera_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/camera.svg"

        # === Animation for Lecture Line 1 ===
        # Display a 2D ant icon [Asset: ...ant.svg] inside a white #FFFFFF circle.
        self.lecture[0].set_color(color_1)
        
        # Create boundary circle (white)
        boundary_circle = Circle(radius=1.5, color=WHITE)
        self.place_in_area(boundary_circle, "B2", "E5")
        
        # Load and place ant (centered in the circle)
        ant = SVGMobject(ant_asset).set_color(GRAY_B)
        self.place_in_area(ant, "B2", "E5", scale_factor=0.8) # Fixed position per Issue 60
        
        self.play(
            Create(boundary_circle),
            FadeIn(ant),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The ant [Asset: ...ant.svg] moves against the circle boundary which flashes red #FF0000.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_2)
        
        # Move ant to the boundary (edge of circle)
        # The circle is centered at (3.0, -0.3) with radius 1.5. 
        # Move it to the right edge (4.5, -0.3)
        edge_pos = self.grid["C5"] + 0.3 * RIGHT
        
        self.play(
            ant.animate.move_to(edge_pos),
            run_time=1.5
        )
        
        # Flash circle red
        self.play(
            boundary_circle.animate.set_color(color_2).set_stroke(width=10),
            run_time=0.4
        )
        self.play(
            boundary_circle.animate.set_color(WHITE).set_stroke(width=4),
            run_time=0.4
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The camera [Asset: ...camera.svg] perspective shifts as the ant [Asset: ...ant.svg] is lifted over the boundary.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_3)
        
        # Add camera icon to indicate shift
        camera_icon = SVGMobject(camera_asset).set_color(color_3)
        self.place_at_grid(camera_icon, "A6", scale_factor=0.6)
        
        # Label for success
        success_label = Text("Dimensional Shift", font_size=20, color=color_3)
        self.place_in_area(success_label, "E5", "E6", scale_factor=0.7) # Fixed position per Issue 60
        
        # Animate lifting shift
        self.play(FadeIn(camera_icon, shift=LEFT))
        
        # Lift ant (scale up then down) and move to outside (Col 6)
        outside_pos = self.grid["C6"]
        
        self.play(
            ant.animate.scale(1.5).move_to(self.grid["B5"]),
            boundary_circle.animate.set_stroke(opacity=0.3),
            run_time=1
        )
        self.play(
            ant.animate.scale(1/1.5).move_to(outside_pos),
            Write(success_label),
            run_time=1
        )
        
        self.wait(2)
