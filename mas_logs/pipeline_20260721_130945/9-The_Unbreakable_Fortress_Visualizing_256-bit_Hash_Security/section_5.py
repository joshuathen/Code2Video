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

class Section5Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "The Laws of Physics: The Ultimate Security Guard"
        lecture_lines = [
            "Security is ultimately guarded by the laws of physics.",
            "Flipping a digital bit requires a minimum amount of energy.",
            "To guess a hash, you need massive power.",
            "This computation would require more energy than the Sun.",
            "Such heat would literally boil the Earth's oceans away."
        ]
        
        # Setup layout
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display 'Energy' (#FFA500) and 'Information' (#00BFFF) with an equals sign.
        self.lecture[0].set_color(YELLOW)
        energy_txt = Text("Energy", color="#FFA500")
        equals_txt = MathTex("=", color=WHITE)
        info_txt = Text("Information", color="#00BFFF")
        
        eq_group = VGroup(energy_txt, equals_txt, info_txt).arrange(RIGHT, buff=0.3)
        self.place_in_area(eq_group, "B2", "B5", scale_factor=0.8)
        
        self.play(FadeIn(eq_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Show a bit flipping 0 to 1 with a heat pulse (#FF4500) radiating.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        bit_box = Square(side_length=1.2, color=WHITE)
        bit_val = Text("0", font_size=48)
        bit_group = VGroup(bit_box, bit_val)
        self.place_at_grid(bit_group, "C3", scale_factor=0.8)
        
        # Heat pulse setup - prepared but not in scene yet
        pulses = VGroup(*[Circle(radius=0.1, color="#FF4500", stroke_width=2) for _ in range(3)])
        for p in pulses:
            p.move_to(self.grid["C3"])
            
        new_bit_val = Text("1", font_size=48).move_to(bit_val)
        
        self.play(
            FadeOut(eq_group),
            FadeIn(bit_group)
        )
        self.wait(1)
        
        self.add(pulses)
        self.play(
            Transform(bit_val, new_bit_val),
            *[p.animate.scale(15).set_stroke(opacity=0) for p in pulses],
            run_time=2,
            rate_func=linear
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Show the Sun (#FDB813) next to an energy bar that quickly overflows.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        sun = Circle(radius=0.6, color="#FDB813", fill_opacity=1)
        sun_glow = Circle(radius=0.75, color="#FDB813", fill_opacity=0.2)
        sun_group = VGroup(sun, sun_glow)
        # Issue 45: Move sun_group to 'D3'
        self.place_at_grid(sun_group, 'D3', scale_factor=0.8)
        
        energy_bar_bg = Rectangle(width=3, height=0.4, color=WHITE)
        energy_bar_fill = Rectangle(width=0.01, height=0.4, color="#FFA500", fill_opacity=0.8).align_to(energy_bar_bg, LEFT)
        energy_label = Text("Hash Energy Cost", font_size=20)
        
        bar_group = VGroup(energy_bar_bg, energy_bar_fill, energy_label)
        energy_label.next_to(energy_bar_bg, UP, buff=0.1)
        # Issue 45: Move bar_group to 'D4'-'D6'
        self.place_in_area(bar_group, 'D4', 'D6', scale_factor=0.8)
        
        self.play(
            FadeOut(bit_group),
            FadeOut(pulses), # Cleaning up remaining pulse objects
            FadeIn(sun_group),
            FadeIn(bar_group)
        )
        
        # Filling the bar
        self.play(
            energy_bar_fill.animate.stretch_to_fit_width(3).align_to(energy_bar_bg, LEFT),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This computation would require more energy than the Sun.
        # An explosion of white and blue light (#FFFFFF, #0000FF) fills the screen.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Overflow
        self.play(
            energy_bar_fill.animate.stretch_to_fit_width(4.5).set_color(RED).align_to(energy_bar_bg, LEFT),
            run_time=1
        )
        
        explosion_particles = VGroup(*[
            Dot(point=sun.get_center(), color=c, radius=0.12) 
            for c in ["#FFFFFF", "#0000FF"] 
            for _ in range(20)
        ])
        
        self.add(explosion_particles)
        self.play(
            *[p.animate.shift(np.array([np.cos(i*0.15), np.sin(i*0.15), 0]) * 5).set_opacity(0) 
              for i, p in enumerate(explosion_particles)],
            sun_group.animate.scale(2.5).set_color(WHITE).set_opacity(0),
            FadeOut(bar_group),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Such heat would literally boil the Earth's oceans away.
        # A blue Earth sphere turns steam-white (#F0F8FF) and evaporates.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        earth_core = Circle(radius=0.8, color=BLUE, fill_opacity=1)
        # Simplified landmass
        land = VGroup(
            Triangle(color=GREEN, fill_opacity=1).scale(0.3).move_to(earth_core.get_center() + UP*0.2 + LEFT*0.1),
            Circle(radius=0.2, color=GREEN, fill_opacity=1).move_to(earth_core.get_center() + DOWN*0.2 + RIGHT*0.3)
        )
        earth = VGroup(earth_core, land)
        # Issue 45: Move 'earth' to 'E3'-'E5'
        self.place_in_area(earth, 'E3', 'E5', scale_factor=0.8)
        
        self.play(FadeIn(earth))
        self.wait(1)
        
        # Steam dots relative to the now-placed earth_core
        steam = VGroup(*[
            Dot(point=earth_core.get_center() + np.array([np.random.uniform(-0.6, 0.6), np.random.uniform(-0.6, 0.6), 0]), 
                color="#F0F8FF", radius=0.1) 
            for _ in range(30)
        ])
        
        self.add(steam)
        self.play(
            earth_core.animate.set_color("#F0F8FF").set_opacity(0.4),
            land.animate.set_opacity(0),
            *[s.animate.shift(UP*2.5 + np.array([np.random.uniform(-1.5, 1.5), 0, 0])).set_opacity(0) 
              for s in steam],
            run_time=4
        )
        
        self.play(FadeOut(earth), FadeOut(explosion_particles))
        self.wait(3)
