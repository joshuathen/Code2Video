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
        self.setup_layout("Summary and Real-World Application", [
            "Matrices are the engine behind modern computer graphics.",
            "Games use linear transformations to render 3D worlds.",
            "Dynamic matrix values move characters in real-time."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display summary bullet points of key concepts #FFFFFF.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer, 'B3', scale_factor=0.5)
        self.play(FadeIn(computer))
        
        # === Animation for Lecture Line 2 ===
        # Show animated example of a simple 2D scaling application #FFFF00.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        square = Square(side_length=1, color=BLUE, fill_opacity=0.5)
        self.place_at_grid(square, 'C3', scale_factor=0.7)
        label = Text("Square", font_size=20)
        self.place_at_grid(label, 'C4', scale_factor=0.5)
        self.add(square, label)
        self.play(square.animate.scale(1.5), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        # Final view showing the transformation process in real-world context #00FFFF.
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        character = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png")
        text_3d = Text("3D Render Pipeline", font_size=24, color=WHITE)
        group_3d_assets = Group(character, text_3d).arrange(DOWN)
        self.place_in_area(group_3d_assets, 'C3', 'D4', scale_factor=0.8)
        self.play(FadeIn(group_3d_assets))
        self.wait(2)
