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
        self.setup_layout("Prerequisite: The Static Slope", ["Slope is the ratio of rise over run.", "Linear paths have a constant steepness.", "Rise over run defines a static slope."])
        
        # Initialize objects
        # Asset usage: road.svg and bicycle.svg
        road = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/road.svg")
        bicycle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bicycle.svg")
        
        # Use road to represent the path, scaled to fit in area B3-F6
        self.place_in_area(road, "B3", "F6", scale_factor=1.5)
        
        # Define a line on top of the road area for movement
        line = Line(self.grid["E5"] + LEFT*1.5, self.grid["C3"] + RIGHT*1.5, color=WHITE)
        
        label_m = MathTex(r"m = \frac{\Delta y}{\Delta x}", color="#FFD700")
        self.place_at_grid(label_m, "D6", scale_factor=1.0) # Fixed via issue 22
        
        path_point = ValueTracker(0)
        
        def update_bicycle(b):
            b.move_to(line.point_from_proportion(path_point.get_value()))
            
        bicycle.add_updater(update_bicycle)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(road), FadeIn(line))
        self.lecture[0].set_color("#FFD700")
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(Write(label_m))
        self.lecture[1].set_color("#FFD700")
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.add(bicycle)
        self.play(path_point.animate.set_value(1), run_time=2)
        self.play(path_point.animate.set_value(0), run_time=2)
        self.lecture[2].set_color("#FFD700")
        self.wait(0.5)
