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
        self.setup_layout("Summary and Geometric Intuition", [
            "High-dimensional space is remarkably hollow and spiky.",
            "Our 3D intuition fails at higher dimensions.",
            "Geometry behaves differently in these strange realms."
        ])
        
        cube_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        # Create a stylized wireframe representation of a high-dimensional object
        # Using the cube asset structure
        cube_wire = SVGMobject(cube_path).set_stroke(color="#00FF00", width=2)
        self.place_at_grid(cube_wire, 'C3', scale_factor=1.5)
        # Add some "spikes"
        spikes = VGroup(*[Line(ORIGIN, UP*0.5).rotate(i * 360/12 * DEGREES).shift(cube_wire.get_center()) for i in range(12)])
        spikes.set_color("#00FF00")
        self.play(FadeIn(cube_wire), Create(spikes))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        # Animate a 3D cube collapsing into a distorted, complex shape
        cube_obj = SVGMobject(cube_path).set_fill(color="#FF00FF", opacity=0.5)
        self.place_at_grid(cube_obj, 'C3', scale_factor=1.5)
        self.play(FadeOut(cube_wire), FadeOut(spikes), FadeIn(cube_obj))
        self.play(cube_obj.animate.scale(0.5).set_color("#FF00FF").stretch_to_fit_width(0.5).stretch_to_fit_height(1.5))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        # Display a closing text summary
        summary = Text("Key Takeaway: High-D is counterintuitive", font_size=24, color="#FFFFFF")
        icon = SVGMobject(cube_path).scale(0.2).set_color("#FFFFFF")
        summary_group = VGroup(summary, icon).arrange(RIGHT)
        self.place_at_grid(summary_group, 'E3', scale_factor=1.0)
        self.play(FadeIn(summary_group))
        self.wait(2)
