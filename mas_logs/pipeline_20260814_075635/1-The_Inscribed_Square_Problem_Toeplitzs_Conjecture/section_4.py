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
        self.setup_layout("Application & Conclusion", [
            "The Square Peg Problem remains fascinating.", 
            "Topology proves existence even without coordinates.", 
            "A perfect fit is mathematically guaranteed."
        ])
        
        # Define objects
        # Asset integration: vehicle.svg and wall.svg
        vehicle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vehicle.svg", color="#00CED1")
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color=WHITE)
        
        touch_points = VGroup(*[Dot(color="#FF1493", radius=0.08) for _ in range(4)])
        label = Text("Topological Existence Guaranteed", font_size=20, color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00CED1")
        self.place_in_area(wall, "B2", "D5", scale_factor=0.8)
        self.place_at_grid(vehicle, "C3", scale_factor=0.6)
        self.play(Create(wall), Create(vehicle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF1493")
        # Ensure points are relative to vehicle points
        for i, dot in enumerate(touch_points):
            dot.move_to(vehicle.get_center())
        self.play(FadeIn(touch_points))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        self.place_at_grid(label, "A5", scale_factor=0.7)
        self.play(Write(label))
        self.wait(2)
