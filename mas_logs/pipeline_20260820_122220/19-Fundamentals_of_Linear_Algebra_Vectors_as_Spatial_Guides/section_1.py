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
        lecture_lines = [
            "Vectors are directed segments in 2D space.",
            "They represent magnitude and specific direction.",
            "Think of them as instructions for movement."
        ]
        self.setup_layout("What is a Vector? (Visualizing Movement)", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Create a 2D arrow representing a vector from origin #FFD700
        self.lecture[0].set_color("#FFD700")
        
        # Load asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, 'B2', scale_factor=0.6)
        
        axes = Axes(x_range=[0, 5], y_range=[0, 5], axis_config={"include_numbers": False}).scale(0.4)
        self.place_in_area(axes, 'B2', 'D5', scale_factor=1.2)
        
        vector = Arrow(axes.c2p(0, 0), axes.c2p(3, 2), color="#FFD700")
        self.add(vector, compass)

        # === Animation for Lecture Line 2 ===
        # Show vector movement animation with trail #FFFFFF
        self.lecture[1].set_color("#FFFFFF")
        trail = TracedPath(vector.get_end, stroke_color=WHITE, stroke_width=2)
        self.add(trail)
        self.play(Create(vector), run_time=2)

        # === Animation for Lecture Line 3 ===
        # Highlight vector components X and Y #FF4500
        self.lecture[2].set_color("#FF4500")
        
        # Load asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        self.place_at_grid(map_icon, 'F6', scale_factor=0.4)
        self.add(map_icon)
        
        x_line = DashedLine(axes.c2p(0, 0), axes.c2p(3, 0), color="#FF4500")
        y_line = DashedLine(axes.c2p(3, 0), axes.c2p(3, 2), color="#FF4500")
        
        x_label = MathTex("x=3", color="#FF4500", font_size=24)
        y_label = MathTex("y=2", color="#FF4500", font_size=24)
        
        # Applying requested position fixes
        self.place_at_grid(x_label, 'E4', scale_factor=0.9)
        x_label.next_to(x_line, DOWN)
        
        self.place_at_grid(y_label, 'C6', scale_factor=0.9)
        y_label.next_to(y_line, RIGHT)
        
        self.play(Create(x_line), Write(x_label), Create(y_line), Write(y_label), run_time=2)
        self.wait(2)
