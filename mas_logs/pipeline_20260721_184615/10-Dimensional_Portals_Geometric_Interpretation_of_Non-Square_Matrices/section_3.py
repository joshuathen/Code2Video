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
        # Data from shared state: title and lecture lines
        title_text = "The 'Span' Perspective"
        lecture_lines = [
            "A 3x2 matrix cannot fill the entire 3D volume.",
            "The output is trapped on a 2D slice.",
            "This slice is the matrix's column space."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Maintain 3D view with tilted plane #555555. 
        # Highlight it with a flashing SurroundingRectangle #FFFF00.
        
        # Create a tilted plane to represent a 2D subspace in 3D space
        plane = Rectangle(width=3.5, height=2.5, color="#555555", fill_opacity=0.6)
        plane.apply_matrix([[1, 0.3, 0], [0, 1, 0], [0, 0, 1]]).rotate(15 * DEGREES)
        # Position in visual area (Right side)
        # Resolved Issue 29: Use 'E5' instead of 'E6'
        self.place_in_area(plane, "B2", "E5", scale_factor=1.0)
        
        self.lecture[0].set_color("#FFFF00")
        self.play(Create(plane))
        self.wait(1.5)
        
        # Highlight animation
        highlight_rect = SurroundingRectangle(plane, color="#FFFF00", buff=0.1)
        self.play(Create(highlight_rect))
        # Lesson L004: Use 'Indicate'
        self.play(Indicate(highlight_rect, color="#FFFF00"))
        self.play(FadeOut(highlight_rect))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Show a 2D circular shadow #333333 being cast by a 3D sphere onto the tilted plane.
        
        # Resolved Issue 19: Use Asset for sphere
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere.set_color(WHITE)
        # Resolved Issue 27: Move sphere to 'B5'
        self.place_at_grid(sphere, "B5", scale_factor=0.5)
        
        # Shadow projected on the plane
        shadow = Ellipse(width=0.6, height=0.25, color="#333333", fill_opacity=0.9, stroke_width=0)
        self.place_at_grid(shadow, "D4") # Positioned approximately at the plane's center
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFFFF")
        
        self.play(FadeIn(sphere), FadeIn(shadow))
        self.wait(1.5)
        
        # Dynamic movement
        self.play(
            sphere.animate.shift(RIGHT * 0.8 + UP * 0.2),
            shadow.animate.shift(RIGHT * 0.5 + DOWN * 0.1),
            run_time=2.5
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Label the tilted plane 'Column Space' #FFFF00. 
        # Fade out the rest of 3D space.
        
        label = Text("Column Space", font_size=24, color="#FFFF00")
        # Resolved Issue 28: Move label to 'E5'
        self.place_at_grid(label, "E5", scale_factor=0.8)
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        self.play(Write(label))
        self.wait(1.5)
        
        # Fading out non-essential elements to emphasize the Column Space
        self.play(FadeOut(sphere), FadeOut(shadow))
        self.wait(2.0)
