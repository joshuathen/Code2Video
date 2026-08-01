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
        # Mandatory setup_layout call with the specific required lecture lines
        self.setup_layout("The Secret: Tracking the Basis Vectors", [
            "We only need to track i-hat and j-hat's movements.",
            "Rotating ninety degrees moves them to new landing spots.",
            "These two coordinates fully define how space has warped."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color("#FF7F7F")) # Light Red
        
        # Create coordinate system geometry
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=3.5,
            background_line_style={"stroke_opacity": 0.4}
        )
        
        # Basis vectors relative to plane
        i_hat = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(1, 0),
            buff=0,
            color="#FF0000",
            stroke_width=6
        )
        j_hat = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(0, 1),
            buff=0,
            color="#00FF00",
            stroke_width=6
        )
        
        # Geometry group for transformation
        grid_geom = VGroup(plane, i_hat, j_hat)
        self.place_in_area(grid_geom, 'B2', 'E5')
        
        # Initial labels for i and j (Text used to avoid latex dependency issues)
        i_label = Text("i", color="#FF0000", font_size=32)
        j_label = Text("j", color="#00FF00", font_size=32)
        self.place_at_grid(i_label, 'C5')
        self.place_at_grid(j_label, 'B4')
        
        self.play(
            Create(plane), 
            GrowArrow(i_hat), 
            GrowArrow(j_hat), 
            FadeIn(i_label), 
            FadeIn(j_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(self.lecture[1].animate.set_color("#FFFF7F")) # Light Yellow
        
        # Rotate the entire grid 90 degrees counter-clockwise
        origin_point = plane.coords_to_point(0, 0)
        self.play(
            Rotate(grid_geom, angle=90*DEGREES, about_point=origin_point),
            FadeOut(i_label),
            FadeOut(j_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(self.lecture[2].animate.set_color("#7FFF7F")) # Light Green
        
        # New basis vector position labels
        self.i_hat_new_text = Text("New i-hat: (0, 1)", font_size=20, color="#FF0000")
        self.j_hat_new_text = Text("New j-hat: (-1, 0)", font_size=20, color="#00FF00")
        
        # Fix for Issue 37: Relocate multi-word i_hat_new_text to an area to avoid overlap
        self.place_in_area(self.i_hat_new_text, 'A4', 'A5', scale_factor=0.6)
        
        # Fix for Issue 38: Relocate multi-word j_hat_new_text to an area to avoid clipping
        self.place_in_area(self.j_hat_new_text, 'D1', 'E1', scale_factor=0.6)
        
        self.play(Write(self.i_hat_new_text), Write(self.j_hat_new_text))
        
        # Demonstrate how another vector (1,1) followed the transformation
        # Unit length in scene coordinates = 3.5 / 4 = 0.875
        unit_len = 0.875
        # After 90 deg CCW rotation, (1,1) moves to (-1,1) relative to origin
        v_transformed_end = origin_point + np.array([-unit_len, unit_len, 0])
        
        v_transformed = Arrow(
            start=origin_point,
            end=v_transformed_end,
            buff=0,
            color=WHITE,
            stroke_width=4
        )
        self.v_label = Text("(1,1) follows", font_size=24, color=WHITE)
        
        # Fix for Issue 39: Position v_label at A2 for better spacing
        self.place_at_grid(self.v_label, 'A2', scale_factor=0.8)
        
        self.play(GrowArrow(v_transformed), FadeIn(self.v_label))
        self.wait(2)
