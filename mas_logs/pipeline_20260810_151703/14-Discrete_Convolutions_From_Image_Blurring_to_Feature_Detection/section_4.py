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
        self.setup_layout("Application: The Edge Detector", [
            "Specific filters can detect edges.",
            "The Sobel operator highlights intensity changes.",
            "Images transform into structural sketches."
        ])
        
        # Asset loading
        img_camera = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        img_photograph = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        kernel = Matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                         h_buff=1.0, v_buff=0.5).set_color("#00FF00")
        self.place_at_grid(kernel, 'C3', scale_factor=0.7) # Issue 30
        self.play(Create(kernel))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        # Representing an intensity grid
        grid_viz = VGroup(*[Square(side_length=0.5).set_stroke(WHITE, 1) for _ in range(9)])
        grid_viz.arrange_in_grid(3, 3, buff=0)
        self.place_at_grid(grid_viz, 'D3', scale_factor=0.9) # Issue 31
        
        self.play(FadeIn(grid_viz))
        self.play(kernel.animate.move_to(grid_viz.get_center()))
        self.play(grid_viz.animate.set_color("#FFFF00"))
        
        # Sliding Sobel kernel over camera icon (Asset integration)
        self.place_at_grid(img_camera, 'D3', scale_factor=0.5)
        self.play(FadeIn(img_camera))
        self.play(kernel.animate.shift(RIGHT * 0.5))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        result_text = Text("Edge Map", font_size=24, color="#00FFFF")
        self.place_at_grid(result_text, 'C4', scale_factor=0.8) # Issue 32
        
        # Comparison (Asset integration)
        self.place_at_grid(img_camera, 'D3', scale_factor=0.4)
        self.place_at_grid(img_photograph, 'D4', scale_factor=0.4)
        
        self.play(Write(result_text))
        self.play(FadeIn(img_photograph))
        self.wait(2)
