from manim import *
import numpy as np

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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application and Future Tech", [
            "Rainbow holograms provide security on cards.",
            "Diffraction enables high-density data storage.",
            "Holography advances modern medical imaging."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Goal: Show a holographic eagle [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/eagle.svg] (#D4AF37) shifting in rainbow colors on a card.
        self.lecture[0].set_color("#FFFF00") # Yellow
        
        card = RoundedRectangle(height=3, width=5, corner_radius=0.2, color="#8E8E8E")
        self.place_in_area(card, "B3", "E5", scale_factor=0.6)
        
        # Load Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/eagle.svg]
        eagle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eagle.svg")
        eagle.set_color("#D4AF37")
        eagle.set_fill("#D4AF37", opacity=0.8)
        self.place_at_grid(eagle, "C4", scale_factor=1.5)
        
        self.play(Create(card), DrawBorderThenFill(eagle))
        
        # Rainbow shifting effect
        rainbow_colors = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"]
        for color in rainbow_colors:
            self.play(eagle.animate.set_color(color), run_time=0.2)
        self.play(eagle.animate.set_color("#D4AF37"), run_time=0.2)
        
        self.wait(1)
        self.play(FadeOut(card), FadeOut(eagle))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Goal: Show white light beam (#FFFFFF) splitting into a spectrum.
        self.lecture[1].set_color("#00FFFF") # Cyan
        
        prism = Triangle(color="#FFFFFF").scale(0.5)
        self.place_at_grid(prism, "C4")
        
        # White light beam coming from C1 to the prism's left side
        white_light = Line(start=self.grid["C1"], end=prism.get_left(), color="#FFFFFF", stroke_width=4)
        
        colors = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082"]
        spectrum_lines = VGroup()
        for i, color in enumerate(colors):
            # Fan out from prism's right side
            target_pos = prism.get_right() + np.array([2.0, (i - 2.5) * 0.4, 0])
            spectrum_lines.add(Line(start=prism.get_right(), end=target_pos, color=color, stroke_width=3))
        
        self.play(Create(white_light))
        self.play(Create(prism))
        self.play(Create(spectrum_lines))
        
        self.wait(1)
        self.play(FadeOut(white_light), FadeOut(prism), FadeOut(spectrum_lines))
        self.lecture[1].set_color(WHITE)

        # === Animation for Lecture Line 3 ===
        # Goal: Display a 3D data cube (#0000FF) with scanning lasers.
        self.lecture[2].set_color("#00FF00") # Green
        
        cube_color = "#0000FF"
        front_face = Square(side_length=1.5, color=cube_color, fill_opacity=0.2)
        back_face = Square(side_length=1.5, color=cube_color, fill_opacity=0.1).shift(0.5*UR)
        
        edges = VGroup(
            Line(front_face.get_corner(UL), back_face.get_corner(UL), color=cube_color),
            Line(front_face.get_corner(UR), back_face.get_corner(UR), color=cube_color),
            Line(front_face.get_corner(DL), back_face.get_corner(DL), color=cube_color),
            Line(front_face.get_corner(DR), back_face.get_corner(DR), color=cube_color),
        )
        cube_group = VGroup(front_face, back_face, edges)
        self.place_in_area(cube_group, "C4", "E6", scale_factor=1.0)
        
        # Scanning laser with ValueTracker and Updater (Persistent Mobject)
        laser_tracker = ValueTracker(0)
        
        # Initialize laser at the left edge of the cube group
        laser_line = Line(
            cube_group.get_left() + DOWN*1.2,
            cube_group.get_left() + UP*1.2,
            color="#FF0000",
            stroke_width=4
        )
        
        def update_laser(mobj):
            x_offset = laser_tracker.get_value()
            new_start = cube_group.get_left() + np.array([x_offset, -1.2, 0])
            new_end = cube_group.get_left() + np.array([x_offset, 1.2, 0])
            mobj.put_start_and_end_on(new_start, new_end)
            
        laser_line.add_updater(update_laser)
        
        self.play(Create(cube_group))
        self.add(laser_line)
        
        # Scan across
        self.play(laser_tracker.animate.set_value(2.2), run_time=2, rate_func=linear)
        self.play(laser_tracker.animate.set_value(0.0), run_time=2, rate_func=linear)
        
        self.wait(2)
        self.play(FadeOut(cube_group), FadeOut(laser_line))
        self.lecture[2].set_color(WHITE)
